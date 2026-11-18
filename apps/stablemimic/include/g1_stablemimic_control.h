#pragma once

#include "g1_hardware_limits.h"
#include "g1_stablemimic_policy.h"

namespace g1_stablemimic {

struct HardwareSettings {
    g1_hardware::LimitsConfig limits;
    std::string base_topic, standing_reference;
    double estimate_timeout, max_skew, ramp_seconds, position_tolerance;
    double entry_joint_speed, entry_pelvis_speed, entry_gyro;
    Joints effort, velocity;

    explicit HardwareSettings(const YAML::Node& root);
};

void CheckMeasuredPositions(const HardwareSettings& settings, const Joints& position);
bool Resting(const HardwareSettings& settings, const State& state);
float RampWeight(const HardwareSettings& settings, double elapsed_seconds);

struct Actuation {
    std::array<float, kJointCount> position{}, kp{}, kd{}, torque{};
};

Actuation ComputeActuation(const HardwareSettings& settings, const DriveParameters& drive,
                           const State& state, const Joints* target, double weight);

}  // namespace g1_stablemimic
