#pragma once

#include "g1_getup_mimic_policy.h"

#include "g1_control_targets.h"
#include "g1_hardware_limits.h"
#include "g1_joint_mapping.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace g1_getup_mimic {

enum class ActuationMode { DrivePd, HostTorque };

inline ActuationMode ParseActuationMode(const std::string& name) {
    if (name == "drive_pd") return ActuationMode::DrivePd;
    if (name == "host_torque") return ActuationMode::HostTorque;
    throw std::invalid_argument("GetUpMimic actuation must be drive_pd or host_torque, got " + name);
}

inline const char* ActuationName(ActuationMode mode) {
    return mode == ActuationMode::DrivePd ? "drive_pd" : "host_torque";
}

struct HardwareSettings {
    g1_hardware::LimitsConfig limits;
    Joints effort;
    ActuationMode actuation = ActuationMode::DrivePd;
    double ramp_seconds;
    double timeout_seconds;
    double entry_joint_speed;
    double entry_gyro_speed;
    double position_tolerance;

    explicit HardwareSettings(const YAML::Node& root, std::optional<ActuationMode> actuation_override = std::nullopt)
        : limits(root, /*use_tilt_limit=*/false),
          effort(g1_config::ReadArray<float, kJointCount>(root["hardware"]["effort_limit"])),
          ramp_seconds(root["hardware"]["getup_ramp_seconds"].as<double>()),
          timeout_seconds(root["hardware"]["getup_timeout_seconds"].as<double>()),
          entry_joint_speed(root["hardware"]["entry_max_joint_speed_rad_s"].as<double>()),
          entry_gyro_speed(root["hardware"]["entry_max_gyro_rad_s"].as<double>()),
          position_tolerance(root["hardware"]["position_tolerance_rad"].as<double>()) {
        const auto configured_actuation = root["hardware"]["actuation"];
        if (actuation_override) actuation = *actuation_override;
        else if (configured_actuation) actuation = ParseActuationMode(configured_actuation.as<std::string>());
        for (double value : {ramp_seconds, timeout_seconds, entry_joint_speed, entry_gyro_speed, position_tolerance}) {
            if (!std::isfinite(value) || value <= 0) throw std::runtime_error("Invalid GetUpMimic hardware setting");
        }
        if (!limits.limit_target_position || timeout_seconds <= ramp_seconds ||
            std::any_of(effort.begin(), effort.end(), [](float value) { return value <= 0; })) {
            throw std::runtime_error("GetUpMimic hardware needs positive effort limits and enabled position limits");
        }
    }
};

inline void CheckMeasuredPositions(const HardwareSettings& settings, const State& state) {
    for (int joint = 0; joint < kJointCount; ++joint) {
        const auto q = state.position[joint];
        if (q < settings.limits.lower[joint] - settings.position_tolerance ||
            q > settings.limits.upper[joint] + settings.position_tolerance) {
            throw std::runtime_error("Measured joint position outside limits: " + g1_joints::kDdsNames[joint]
                                     + " at " + std::to_string(q) + " rad");
        }
    }
}

inline float GyroSpeed(const State& state) {
    const auto& [x, y, z] = state.pelvis_angular_velocity;
    return std::hypot(x, y, z);
}

inline bool ReadyToGetUp(const HardwareSettings& settings, const State& state) {
    return GyroSpeed(state) <= settings.entry_gyro_speed &&
           std::all_of(state.velocity.begin(), state.velocity.end(),
                       [&](float speed) { return std::abs(speed) <= settings.entry_joint_speed; });
}

inline std::string EntryMotionBlockReason(const HardwareSettings& settings, const State& state) {
    if (GyroSpeed(state) > settings.entry_gyro_speed)
        return "Pelvis angular speed=" + std::to_string(GyroSpeed(state)) +
            " rad/s exceeds " + std::to_string(settings.entry_gyro_speed) + " rad/s";
    for (int j = 0; j < kJointCount; ++j) {
        if (std::abs(state.velocity[j]) > settings.entry_joint_speed)
            return "Joint still moving: " + g1_joints::kDdsNames[j] +
                " speed=" + std::to_string(std::abs(state.velocity[j])) +
                " rad/s exceeds " + std::to_string(settings.entry_joint_speed) + " rad/s";
    }
    return {};
}

inline bool Standing(const State& state) {
    // Use the available pelvis attitude and knee states in the absence of root-height ground truth.
    return -ProjectedGravity(state.pelvis_quaternion)[2] > .9f && GyroSpeed(state) < .6f &&
           std::abs(state.position[3]) < 1.f && std::abs(state.position[9]) < 1.f &&
           std::abs(state.velocity[3]) < 1.f && std::abs(state.velocity[9]) < 1.f;
}

struct Actuation {
    MotorTargets applied;
    Joints torque{};
};

inline Actuation ComputeActuation(const HardwareSettings& settings, const State& state,
                                 const MotorTargets* targets, float weight) {
    Actuation command{};
    for (int joint = 0; joint < kJointCount; ++joint) {
        const auto q = state.position[joint];
        const auto dq = state.velocity[joint];
        const auto low = settings.limits.lower[joint];
        const auto high = settings.limits.upper[joint];
        const float target = std::clamp(targets ? targets->position[joint] : q, low, high);
        const float kp = targets ? weight * targets->kp[joint] : 0.f;
        const float kd = targets ? settings.limits.stop_kd + weight * (targets->kd[joint] - settings.limits.stop_kd)
                                 : settings.limits.stop_kd;
        const float estimated_torque = kp * (target - q) - kd * dq;
        if (!std::isfinite(estimated_torque)) throw std::runtime_error("Non-finite GetUpMimic PD torque");
        float torque = std::clamp(estimated_torque, -settings.effort[joint], settings.effort[joint]);
        if ((q <= low && torque < 0) || (q >= high && torque > 0)) torque = 0;
        // Scale both gains when the host estimate hits a limit. The drive evaluates PD on its own feedback.
        float gain_scale = 1.f;
        if (settings.actuation == ActuationMode::DrivePd && torque != estimated_torque) {
            gain_scale = torque / estimated_torque;
        }
        command.applied.position[joint] = target;
        command.applied.kp[joint] = kp * gain_scale;
        command.applied.kd[joint] = kd * gain_scale;
        command.torque[joint] = torque;
    }
    return command;
}

inline float RampWeight(const HardwareSettings& settings, double elapsed_seconds) {
    return g1_control::InterpolateReset(0.f, 1.f, g1_control::ResetProgress(elapsed_seconds, settings.ramp_seconds));
}

class RecoveryProgress {
public:
    enum class Status { Rising, Standing, TimedOut };

    bool IsStanding() const { return standing_; }

    void Reset() {
        stable_since_.reset();
        standing_ = false;
    }

    Status Update(const HardwareSettings& settings, const State& state, double elapsed_seconds) {
        if (standing_) return Status::Standing;
        if (Standing(state)) {
            if (!stable_since_) stable_since_ = elapsed_seconds;
            standing_ = elapsed_seconds - *stable_since_ >= 1.0;
            if (standing_) return Status::Standing;
        } else {
            stable_since_.reset();
        }
        return elapsed_seconds >= settings.timeout_seconds ? Status::TimedOut : Status::Rising;
    }

private:
    std::optional<double> stable_since_;
    bool standing_ = false;
};

}  // namespace g1_getup_mimic
