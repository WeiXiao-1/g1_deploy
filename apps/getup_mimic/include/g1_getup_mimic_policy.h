#pragma once

#include <array>
#include <chrono>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>

namespace g1_getup_mimic {

inline constexpr int kJointCount = 29;
inline constexpr int kHistoryFrames = 8;
inline constexpr int kFrameSize = 3 + 3 + 3 * kJointCount;
inline constexpr int kObservationSize = kFrameSize * kHistoryFrames;
inline constexpr auto kPolicyPeriod = std::chrono::milliseconds(10);

using Joints = std::array<float, kJointCount>;
using Observation = std::array<float, kObservationSize>;

struct State {
    Joints position{}, velocity{};  // DDS motor order, rad and rad/s
    std::array<float, 4> pelvis_quaternion{1, 0, 0, 0};  // w, x, y, z
    std::array<float, 3> pelvis_angular_velocity{};       // pelvis frame, rad/s
};

std::array<float, 3> ProjectedGravity(const std::array<float, 4>& quaternion);

struct MotorTargets {
    Joints position{}, kp{}, kd{};  // DDS motor order
};

class Policy {
public:
    explicit Policy(const std::string& config_path);

    MotorTargets Step(const State& state);
    void Reset();

    // Exposes the model boundary for checking exported observation/action pairs.
    Joints Predict(Observation& observation);

private:
    using Frame = std::array<float, kFrameSize>;
    Frame BuildFrame(const State& state) const;

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "g1_getup_mimic"};
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_, output_name_;
    std::array<int, kJointCount> motor_ids_{};
    Joints default_position_{}, action_scale_{}, last_action_{};
    Joints motor_kp_{}, motor_kd_{};
    std::array<Frame, kHistoryFrames> history_{};
    bool history_ready_ = false;
};

}  // namespace g1_getup_mimic
