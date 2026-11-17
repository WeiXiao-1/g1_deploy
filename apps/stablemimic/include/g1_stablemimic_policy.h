#pragma once

#include <array>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <string>

namespace g1_stablemimic {

inline constexpr auto kPolicyPeriod = std::chrono::milliseconds(20);
inline constexpr int kJointCount = 29;
inline constexpr int kHistoryFrames = 4;
inline constexpr int kGateFrameSize = 6 + 3 * kJointCount;
inline constexpr int kPolicyFrameSize = 4 * kJointCount + 12 + kGateFrameSize;
inline constexpr int kGateSize = kGateFrameSize * kHistoryFrames;
inline constexpr int kPolicySize = kPolicyFrameSize * kHistoryFrames;

using Joints = std::array<double, kJointCount>;
using Vector3 = std::array<double, 3>;
using Rotation = std::array<double, 9>;  // Row-major, body to world.
inline constexpr Rotation kIdentityRotation{1, 0, 0, 0, 1, 0, 0, 0, 1};

struct State {
    Joints position{}, velocity{};  // DDS joint order.
    Rotation pelvis_rotation = kIdentityRotation;
    Vector3 pelvis_angular_velocity_world{}, pelvis_linear_velocity_world{};
    Vector3 torso_position_world{};
    Rotation torso_rotation = kIdentityRotation;
};

struct Reference {
    Joints position{}, velocity{};  // DDS joint order.
    Vector3 torso_position_world{};  // Includes any translation chosen by the reference source.
    Rotation torso_rotation = kIdentityRotation;
};

struct Observations {
    std::array<float, kPolicySize> policy{};
    std::array<float, kGateSize> gate{};
};

struct Result {
    std::array<float, kJointCount> action{};  // Raw ONNX action in policy order.
    std::array<float, 2> gate_weights{};      // Tracking, recovery.
    Joints target{};                         // Scaled target angle in DDS order.
};

struct DriveParameters {
    Joints kp{}, kd{}, armature{}, effort_limit{}, velocity_limit{};  // DDS order.
};

class Policy {
public:
    explicit Policy(const std::string& model_path);
    Result Step(const State& state, const Reference& reference, Observations* observed = nullptr);
    void Reset();
    const DriveParameters& Drive() const { return drive_; }

private:
    struct Frame {
        std::array<float, kPolicyFrameSize> policy{};
        std::array<float, kGateFrameSize> gate{};
    };
    Frame BuildFrame(const State& state, const Reference& reference) const;

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "g1_stablemimic"};
    Ort::Session session_;
    std::array<int, kJointCount> motor_ids_{};
    Joints nominal_{}, scale_{};
    std::optional<float> action_clip_;
    DriveParameters drive_;
    std::array<Frame, kHistoryFrames> history_{};
    std::array<float, kJointCount> last_action_{};
    bool history_ready_ = false;
};

}  // namespace g1_stablemimic
