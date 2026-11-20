#include "g1_getup_mimic_policy.h"

#include "g1_config_utils.h"
#include "g1_joint_mapping.h"
#include "g1_observation_history.h"
#include "g1_onnx_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace g1_getup_mimic {
namespace {

constexpr std::array<int, 5> kTermSizes{3, 3, kJointCount, kJointCount, kJointCount};

bool SupportsSingleBatch(const std::vector<int64_t>& shape, int features) {
    return shape.size() == 2 && (shape[0] == 1 || shape[0] == -1) && shape[1] == features;
}

}  // namespace

PolicyConfig::PolicyConfig(const std::string& config_path, const std::string& expert)
    : path(config_path), root(YAML::LoadFile(path)), model(root) {
    if (expert.empty()) return;

    const auto experts = root["model_test"];
    const auto selected = experts && experts.IsMap() ? experts[expert] : YAML::Node{};
    if (!selected || !selected.IsMap()) throw std::invalid_argument("Unknown model_test expert: " + expert);
    model.reset(selected);
    observation_size = kFrameSize;

    action_clip = selected["action_clip"].as<float>();
    if (!std::isfinite(action_clip) || action_clip <= 0.f) {
        throw std::invalid_argument("Expected a positive finite model_test action_clip for " + expert);
    }
}

Policy::Policy(const PolicyConfig& settings)
    : observation_size_(settings.observation_size), action_clip_(settings.action_clip) {
    const auto& config = settings.root;
    motor_ids_ = g1_joints::FromNames(config["joint_names"].as<std::vector<std::string>>());
    default_position_ = g1_config::ReadArray<float, kJointCount>(config["default_joint_pos"]);
    action_scale_ = g1_config::ReadArray<float, kJointCount>(config["action_scale"]);
    motor_kp_ = g1_joints::ToMotorOrder(g1_config::ReadArray<float, kJointCount>(config["kp"]), motor_ids_);
    motor_kd_ = g1_joints::ToMotorOrder(g1_config::ReadArray<float, kJointCount>(config["kd"]), motor_ids_);
    for (int joint = 0; joint < kJointCount; ++joint) {
        if (action_scale_[joint] <= 0 || motor_kp_[joint] < 0 || motor_kd_[joint] < 0) {
            throw std::runtime_error("Invalid GetUpMimic action scale or PD gain");
        }
    }

    const auto model_path = g1_config::ResolvePath(settings.path, settings.model["policy_path"].as<std::string>());
    auto options = g1_onnx::CpuSessionOptions();
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1 ||
        !SupportsSingleBatch(g1_onnx::TensorShape(*session_, true, 0), observation_size_) ||
        !SupportsSingleBatch(g1_onnx::TensorShape(*session_, false, 0), kJointCount)) {
        throw std::runtime_error("Expected GetUpMimic float32 tensors [batch," + std::to_string(observation_size_)
                                 + "] -> [batch,29]");
    }
    input_name_ = g1_onnx::TensorName(*session_, true, 0);
    output_name_ = g1_onnx::TensorName(*session_, false, 0);
    if (input_name_ != "obs" || output_name_ != "actions") {
        throw std::runtime_error("Expected GetUpMimic tensor names obs and actions");
    }
}

void Policy::Reset() {
    last_action_.fill(0);
    history_ready_ = false;
}

std::array<float, 3> ProjectedGravity(const std::array<float, 4>& quaternion) {
    const auto& [w, x, y, z] = quaternion;
    const double norm_squared = double(w) * w + double(x) * x + double(y) * y + double(z) * z;
    if (!std::isfinite(norm_squared) || norm_squared < 1e-12) {
        throw std::runtime_error("Invalid GetUpMimic pelvis quaternion");
    }

    // Rotate world gravity (0, 0, -1) into the pelvis frame; account for quaternion norm.
    return {
        static_cast<float>(2 * (double(w) * y - double(x) * z) / norm_squared),
        static_cast<float>(-2 * (double(w) * x + double(y) * z) / norm_squared),
        static_cast<float>(2 * (double(x) * x + double(y) * y) / norm_squared - 1),
    };
}

Policy::Frame Policy::BuildFrame(const State& state) const {
    const auto gravity = ProjectedGravity(state.pelvis_quaternion);
    Frame frame{
        state.pelvis_angular_velocity[0], state.pelvis_angular_velocity[1], state.pelvis_angular_velocity[2],
        gravity[0], gravity[1], gravity[2],
    };
    for (int joint = 0; joint < kJointCount; ++joint) {
        const int motor = motor_ids_[joint];
        frame[6 + joint] = state.position[motor] - default_position_[joint];
        frame[6 + kJointCount + joint] = state.velocity[motor];  // Training default joint velocity is zero.
        frame[6 + 2 * kJointCount + joint] = last_action_[joint];
    }
    return frame;
}

Joints Policy::Predict(float* observation, std::size_t size) {
    if (size != static_cast<std::size_t>(observation_size_)) {
        throw std::runtime_error("Invalid GetUpMimic observation size; expected " + std::to_string(observation_size_));
    }
    if (std::any_of(observation, observation + size, [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("Non-finite GetUpMimic observation");
    }
    const g1_onnx::InputTensor input{input_name_.c_str(), observation, size, {1, observation_size_}};
    return g1_onnx::Run<kJointCount>(*session_, {input}, output_name_);
}

MotorTargets Policy::Step(const State& state, trace::PolicySnapshot* snapshot) {
    auto next_history = history_;
    bool next_history_ready = history_ready_;
    auto frame = BuildFrame(state);
    Observation history_observation;
    float* observation = frame.data();
    if (observation_size_ == kObservationSize) {
        g1_inference::PushHistory(next_history, frame, next_history_ready);
        g1_inference::PackTermMajor(next_history, kTermSizes, history_observation.begin());
        observation = history_observation.data();
    }

    // The history model clips inside ONNX. Expert exports return raw actor actions.
    if (snapshot) {
        static_assert(kObservationSize == trace::kMaxObservation);
        snapshot->observation_size = observation_size_;
        std::copy_n(observation, observation_size_, snapshot->observation.begin());
    }
    auto action = Predict(observation, observation_size_);
    if (snapshot) snapshot->onnx_action = action;
    if (action_clip_ > 0.f) {
        for (float& value : action) value = std::clamp(value, -action_clip_, action_clip_);
    }
    MotorTargets result{{}, motor_kp_, motor_kd_};
    for (int joint = 0; joint < kJointCount; ++joint) {
        const float position = default_position_[joint] + action_scale_[joint] * action[joint];
        if (!std::isfinite(position)) {
            throw std::runtime_error("Non-finite GetUpMimic joint target");
        }
        result.position[motor_ids_[joint]] = position;
    }
    history_ = next_history;
    history_ready_ = next_history_ready;
    last_action_ = action;
    if (snapshot) snapshot->applied_action = action;
    return result;
}

}  // namespace g1_getup_mimic
