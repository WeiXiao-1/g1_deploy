#pragma once

#include "g1_getup_mimic_control.h"
#include "g1_config_utils.h"

#include <map>
#include <string>

namespace g1_getup_mimic {

// GetUpMimic-only pose motion. Reuse interpolation/actuation helpers without
// changing DWaQ, model observation offsets or policy action scales.
class PoseReset {
public:
    PoseReset(const PolicyConfig& config, const HardwareSettings& hardware,
              const std::string& selected = "recovered") {
        const auto path = g1_config::ResolvePath(config.path,
            config.root["reset_path"] ? config.root["reset_path"].as<std::string>()
                                      : "g1_getup_mimic_reset.json");
        const auto root = YAML::LoadFile(path);
        const auto ids = g1_joints::FromNames(root["joint_names"].as<std::vector<std::string>>());
        for (const auto& item : root["profiles"]) {
            const auto name = item.first.as<std::string>();
            auto q = g1_joints::ToMotorOrder(
                g1_config::ReadArray<float, kJointCount>(item.second["joint_position"]), ids);
            for (int j = 0; j < kJointCount; ++j) {
                if (!std::isfinite(q[j]) || q[j] < hardware.limits.lower[j] || q[j] > hardware.limits.upper[j]) {
                    throw std::runtime_error("Reset pose outside hardware limits: " + name);
                }
            }
            poses_.emplace(name, q);
        }
        auto positive = [&](const char* key) {
            const double value = root[key].as<double>();
            if (!std::isfinite(value) || value <= 0) throw std::runtime_error(std::string("Invalid reset ") + key);
            return value;
        };
        duration_ = positive("duration_seconds");
        max_speed_ = positive("max_target_speed_rad_s");
        gain_ramp_ = positive("gain_ramp_seconds");
        tolerance_ = positive("position_tolerance_rad");
        settle_speed_ = positive("settle_speed_rad_s");
        if (duration_ < 1 || max_speed_ > 1 || tolerance_ > .25 || gain_ramp_ > duration_) {
            throw std::runtime_error("Reset motion settings outside supported range");
        }
        const auto motor_ids = g1_joints::FromNames(config.root["joint_names"].as<std::vector<std::string>>());
        kp_ = g1_joints::ToMotorOrder(g1_config::ReadArray<float, kJointCount>(config.root["kp"]), motor_ids);
        kd_ = g1_joints::ToMotorOrder(g1_config::ReadArray<float, kJointCount>(config.root["kd"]), motor_ids);
        stop_kd_ = hardware.limits.stop_kd;
        Select(selected);
    }

    void Select(const std::string& name) {
        if (!poses_.count(name)) throw std::invalid_argument("Unknown reset pose: " + name + " (use recovered, recorded, 3 or dwaq-reset)");
        if (active_ && !complete_) throw std::runtime_error("Wait for pose reset before selecting another pose");
        selected_ = name;  // Does not change the target currently being held.
    }
    const std::string& Selected() const { return selected_; }
    bool Active() const { return active_; }
    bool Complete() const { return complete_; }
    double Duration() const { return motion_duration_; }
    void Cancel() { active_ = complete_ = false; }

    std::string EntryBlockReason(const State& state) const {
        if (!active_) return "No active reset pose";
        if (!complete_) return "Wait for reset target interpolation to finish";
        int worst = 0;
        for (int j = 0; j < kJointCount; ++j) {
            if (!std::isfinite(state.position[j]) || !std::isfinite(state.velocity[j]))
                return "Non-finite reset joint state: " + g1_joints::kDdsNames[j];
            if (std::abs(state.position[j] - goal_[j]) > std::abs(state.position[worst] - goal_[worst])) worst = j;
        }
        const float error = std::abs(state.position[worst] - goal_[worst]);
        if (error > tolerance_) {
            return "Reset pose error at " + g1_joints::kDdsNames[worst] +
                ": actual=" + std::to_string(state.position[worst]) +
                " rad, target=" + std::to_string(goal_[worst]) +
                " rad, error=" + std::to_string(error) +
                " rad exceeds " + std::to_string(tolerance_) + " rad";
        }
        for (int j = 0; j < kJointCount; ++j) {
            if (std::abs(state.velocity[j]) > settle_speed_)
                return "Reset joint still moving: " + g1_joints::kDdsNames[j] +
                    " speed=" + std::to_string(std::abs(state.velocity[j])) +
                    " rad/s exceeds " + std::to_string(settle_speed_) + " rad/s";
        }
        return {};
    }

    bool AtGoal(const State& state) const {
        for (int j = 0; j < kJointCount; ++j) {
            if (!std::isfinite(state.position[j]) || !std::isfinite(state.velocity[j]) ||
                std::abs(state.position[j] - goal_[j]) > tolerance_ ||
                std::abs(state.velocity[j]) > settle_speed_) return false;
        }
        return true;
    }

    void Begin(const State& state, double now) {
        start_ = state.position;
        goal_ = poses_.at(selected_);
        motion_duration_ = duration_;
        for (int j = 0; j < kJointCount; ++j) {
            if (!std::isfinite(start_[j])) throw std::runtime_error("Invalid reset start angle");
            // Max derivative of cubic smoothstep is 1.5.
            motion_duration_ = std::max(motion_duration_, 1.5 * std::abs(goal_[j] - start_[j]) / max_speed_);
        }
        started_ = now;
        active_ = true;
        complete_ = false;
    }

    MotorTargets Update(double now) {
        if (!active_) throw std::runtime_error("Pose reset is not active");
        const double elapsed = std::max(0., now - started_);
        const double progress = g1_control::ResetProgress(elapsed, motion_duration_);
        MotorTargets targets{g1_control::InterpolateResetPose(start_, goal_, progress), kp_, kd_};
        const float weight = g1_control::InterpolateReset(0.f, 1.f,
            g1_control::ResetProgress(elapsed, gain_ramp_));
        for (int j = 0; j < kJointCount; ++j) {
            targets.kp[j] *= weight;
            targets.kd[j] = stop_kd_ + weight * (kd_[j] - stop_kd_);
        }
        // Like DWaQ, Holding means interpolation completed; policy entry checks
        // measured position/velocity separately. Keep commanding the fixed goal.
        if (progress >= 1) complete_ = true;
        return targets;
    }

private:
    std::map<std::string, Joints> poses_;
    std::string selected_;
    Joints start_{}, goal_{}, kp_{}, kd_{};
    double duration_{}, max_speed_{}, gain_ramp_{}, tolerance_{}, settle_speed_{};
    double started_ = 0, motion_duration_ = 0;
    float stop_kd_ = 1;
    bool active_ = false, complete_ = false;
};

inline MotorTargets BlendPoseEntry(const MotorTargets& hold, const MotorTargets& policy, float weight) {
    MotorTargets result;
    for (int j = 0; j < kJointCount; ++j) {
        result.position[j] = hold.position[j] + weight * (policy.position[j] - hold.position[j]);
        result.kp[j] = hold.kp[j] + weight * (policy.kp[j] - hold.kp[j]);
        result.kd[j] = hold.kd[j] + weight * (policy.kd[j] - hold.kd[j]);
    }
    return result;
}

}  // namespace g1_getup_mimic
