#include "g1_stablemimic_policy.h"

#include "g1_config_utils.h"
#include "g1_joint_mapping.h"
#include "g1_observation_history.h"
#include "g1_onnx_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace g1_stablemimic {
namespace {

constexpr char kMetadata[] = "stablemimic.mujoco.v1";

void ValidateTensors(Ort::Session& session, bool input, const char* first, int first_size,
                     const char* second, int second_size) {
    const auto count = input ? session.GetInputCount() : session.GetOutputCount();
    if (count != 2) throw std::runtime_error("StableMimic requires two inputs and two outputs");
    bool found_first = false, found_second = false;
    for (std::size_t index = 0; index < count; ++index) {
        const auto name = g1_onnx::TensorName(session, input, index);
        const auto shape = g1_onnx::TensorShape(session, input, index);
        found_first |= name == first && shape == std::vector<int64_t>{1, first_size};
        found_second |= name == second && shape == std::vector<int64_t>{1, second_size};
    }
    if (!found_first || !found_second) throw std::runtime_error("Incompatible StableMimic tensor names or shapes");
}

Vector3 IntoBody(const Rotation& rotation, const Vector3& world) {
    Vector3 result{};
    for (int i = 0; i < 3; ++i) {
        result[i] = rotation[i] * world[0] + rotation[3 + i] * world[1] + rotation[6 + i] * world[2];
    }
    return result;
}

}  // namespace

Policy::Policy(const std::string& model_path)
    : session_(env_, model_path.c_str(), g1_onnx::CpuSessionOptions()) {
    ValidateTensors(session_, true, "policy", kPolicySize, "gate", kGateSize);
    ValidateTensors(session_, false, "action", kJointCount, "gate_weights", 2);
    Ort::AllocatorWithDefaultOptions allocator;
    const auto metadata = session_.GetModelMetadata().LookupCustomMetadataMapAllocated(kMetadata, allocator);
    if (!metadata) throw std::runtime_error("Missing stablemimic.mujoco.v1 ONNX metadata");
    const auto config = YAML::Load(metadata.get());
    motor_ids_ = g1_joints::FromNames(config["joint_names"].as<std::vector<std::string>>());
    nominal_ = g1_config::ReadArray<double, kJointCount>(config["nominal_joint_positions"]);
    scale_ = g1_config::ReadArray<double, kJointCount>(config["action_scale"]);
    if (!config["action_clip"].IsNull()) {
        action_clip_ = config["action_clip"].as<float>();
        if (!std::isfinite(*action_clip_) || *action_clip_ <= 0) {
            throw std::runtime_error("Invalid StableMimic action_clip");
        }
    }
    const auto drive = config["drive"];
    const auto motor_values = [&](const char* key) {
        return g1_joints::ToMotorOrder(g1_config::ReadArray<double, kJointCount>(drive[key]), motor_ids_);
    };
    drive_ = {motor_values("stiffness"), motor_values("damping"), motor_values("armature"),
              motor_values("joint_effort_limit"), motor_values("joint_velocity_limit")};
    for (int i = 0; i < kJointCount; ++i) {
        if (scale_[i] <= 0 || drive_.kp[i] < 0 || drive_.kd[i] < 0 || drive_.armature[i] < 0 ||
            drive_.effort_limit[i] <= 0 || drive_.velocity_limit[i] <= 0) {
            throw std::runtime_error("Invalid StableMimic scale or drive parameters");
        }
    }
}

void Policy::Reset() {
    last_action_.fill(0);
    history_ready_ = false;
}

Policy::Frame Policy::BuildFrame(const State& state, const Reference& reference) const {
    Frame result;
    const auto omega = IntoBody(state.pelvis_rotation, state.pelvis_angular_velocity_world);
    for (int i = 0; i < 3; ++i) {
        result.gate[i] = static_cast<float>(omega[i]);
        result.gate[3 + i] = static_cast<float>(-state.pelvis_rotation[6 + i]);
    }
    for (int i = 0; i < kJointCount; ++i) {
        const int motor = motor_ids_[i];
        result.gate[6 + i] = static_cast<float>(state.position[motor] - nominal_[i]);
        result.gate[6 + kJointCount + i] = static_cast<float>(state.velocity[motor]);
        result.gate[6 + 2 * kJointCount + i] = last_action_[i];
        result.policy[i] = static_cast<float>(reference.position[motor]);
        result.policy[kJointCount + i] = static_cast<float>(reference.velocity[motor]);
    }

    Vector3 delta{};
    for (int i = 0; i < 3; ++i) delta[i] = reference.torso_position_world[i] - state.torso_position_world[i];
    auto output = result.policy.begin() + 2 * kJointCount;
    const auto anchor_position = IntoBody(state.torso_rotation, delta);
    output = std::copy(anchor_position.begin(), anchor_position.end(), output);

    // The reference orientation contributes its first two columns, interleaved by row.
    const auto& rotation = reference.torso_rotation;
    const auto first = IntoBody(state.torso_rotation, {rotation[0], rotation[3], rotation[6]});
    const auto second = IntoBody(state.torso_rotation, {rotation[1], rotation[4], rotation[7]});
    for (int i = 0; i < 3; ++i) {
        *output++ = static_cast<float>(first[i]);
        *output++ = static_cast<float>(second[i]);
    }
    const auto velocity = IntoBody(state.pelvis_rotation, state.pelvis_linear_velocity_world);
    output = std::copy(velocity.begin(), velocity.end(), output);
    std::copy(result.gate.begin(), result.gate.end(), output);
    // The trailing 58 masked get-up reference values remain zero.
    if (std::any_of(result.policy.begin(), result.policy.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("Non-finite StableMimic observation");
    }
    return result;
}

Result Policy::Step(const State& state, const Reference& reference, Observations* observed) {
    auto history = history_;
    bool ready = history_ready_;
    g1_inference::PushHistory(history, BuildFrame(state, reference), ready);
    Observations inputs;
    auto policy = inputs.policy.begin();
    auto gate = inputs.gate.begin();
    for (const auto& frame : history) {
        policy = std::copy(frame.policy.begin(), frame.policy.end(), policy);
        gate = std::copy(frame.gate.begin(), frame.gate.end(), gate);
    }

    const auto outputs = g1_onnx::RunOutputs(session_,
        {{"policy", inputs.policy.data(), inputs.policy.size(), {1, kPolicySize}},
         {"gate", inputs.gate.data(), inputs.gate.size(), {1, kGateSize}}},
        {"action", "gate_weights"});
    Result result{g1_onnx::ReadOutput<kJointCount>(outputs[0]), g1_onnx::ReadOutput<2>(outputs[1]), {}};
    auto clipped = result.action;
    for (int i = 0; i < kJointCount; ++i) {
        if (action_clip_) clipped[i] = std::clamp(clipped[i], -*action_clip_, *action_clip_);
        const double target = nominal_[i] + scale_[i] * clipped[i];
        if (!std::isfinite(target)) throw std::runtime_error("Non-finite StableMimic target angle");
        result.target[motor_ids_[i]] = target;
    }
    last_action_ = clipped;
    history_ = history;
    history_ready_ = ready;
    if (observed) *observed = inputs;
    return result;
}

}  // namespace g1_stablemimic
