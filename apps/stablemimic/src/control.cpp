#include "g1_stablemimic_control.h"

#include "g1_control_targets.h"
#include "g1_joint_mapping.h"

#include <algorithm>
#include <cmath>

namespace g1_stablemimic {
namespace {

double PositiveSetting(const YAML::Node& node, const char* name, double maximum) {
    const auto value = node[name].as<double>();
    if (!std::isfinite(value) || value <= 0 || value > maximum)
        throw std::runtime_error(std::string("Invalid StableMimic hardware setting: ") + name);
    return value;
}

}  // namespace

HardwareSettings::HardwareSettings(const YAML::Node& root) : limits(root, false) {
    const auto hw = root["hardware"];
    base_topic = hw["base_state_topic"].as<std::string>();
    standing_reference = hw["standing_reference"].as<std::string>();
    if (base_topic.empty() || base_topic == "rt/sportmodestate" || base_topic == "rt/odommodestate" ||
        standing_reference.empty())
        throw std::runtime_error("Configure an independent base state topic and a standing reference");
    estimate_timeout = PositiveSetting(hw, "estimate_timeout_ms", 150);
    max_skew = PositiveSetting(hw, "max_state_skew_ms", estimate_timeout);
    ramp_seconds = PositiveSetting(hw, "ramp_seconds", 5);
    position_tolerance = PositiveSetting(hw, "position_tolerance_rad", .10);
    entry_joint_speed = PositiveSetting(hw, "entry_max_joint_speed_rad_s", 5);
    entry_pelvis_speed = PositiveSetting(hw, "entry_max_pelvis_speed_m_s", 2);
    entry_gyro = PositiveSetting(hw, "entry_max_gyro_rad_s", 3);
    effort = g1_config::ReadArray<double, kJointCount>(hw["effort_limit"]);
    velocity = g1_config::ReadArray<double, kJointCount>(hw["velocity_limit"]);
    for (int i = 0; i < kJointCount; ++i)
        if (effort[i] <= 0 || velocity[i] <= 0)
            throw std::runtime_error("Joint effort and speed limits must be positive");
}

void CheckMeasuredPositions(const HardwareSettings& settings, const Joints& position) {
    for (int joint = 0; joint < kJointCount; ++joint) {
        const double q = position[joint];
        if (q < settings.limits.lower[joint] - settings.position_tolerance ||
            q > settings.limits.upper[joint] + settings.position_tolerance)
            throw std::runtime_error("Measured joint outside limits: " + g1_joints::kDdsNames[joint] +
                                     " at " + std::to_string(q) + " rad");
    }
}

bool Resting(const HardwareSettings& settings, const State& state) {
    const auto& omega = state.pelvis_angular_velocity_world;
    const auto& v = state.pelvis_linear_velocity_world;
    return std::hypot(omega[0], omega[1], omega[2]) <= settings.entry_gyro &&
           std::hypot(v[0], v[1], v[2]) <= settings.entry_pelvis_speed &&
           std::all_of(state.velocity.begin(), state.velocity.end(),
                       [&](double dq) { return std::abs(dq) <= settings.entry_joint_speed; });
}

float RampWeight(const HardwareSettings& settings, double elapsed_seconds) {
    return g1_control::InterpolateReset(0.f, 1.f, g1_control::ResetProgress(elapsed_seconds, settings.ramp_seconds));
}

Actuation ComputeActuation(const HardwareSettings& settings, const DriveParameters& drive,
                           const State& state, const Joints* target, double weight) {
    Actuation actuation;
    for (int i = 0; i < kJointCount; ++i) {
        const double q = state.position[i], dq = state.velocity[i];
        const double low = settings.limits.lower[i], high = settings.limits.upper[i];
        const double goal = std::clamp(target ? (*target)[i] : q, low, high);
        const double kp = target ? weight * drive.kp[i] : 0.;
        const double kd = settings.limits.stop_kd + weight * (drive.kd[i] - settings.limits.stop_kd);
        double torque = kp * (goal - q) - kd * dq;
        if (!std::isfinite(torque)) throw std::runtime_error("Invalid StableMimic motor torque");
        // Permit braking at the position or speed boundary.
        const double speed = std::min(settings.velocity[i], drive.velocity_limit[i]);
        if ((torque < 0 && (q <= low || dq <= -speed)) ||
            (torque > 0 && (q >= high || dq >= speed))) torque = 0;
        const double effort = std::min(settings.effort[i], drive.effort_limit[i]);
        actuation.position[i] = static_cast<float>(goal);
        actuation.kp[i] = static_cast<float>(kp);
        actuation.kd[i] = static_cast<float>(kd);
        actuation.torque[i] = static_cast<float>(std::clamp(torque, -effort, effort));
    }
    return actuation;
}

}  // namespace g1_stablemimic
