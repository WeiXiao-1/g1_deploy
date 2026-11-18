#include "g1_stablemimic_control.h"
#include "g1_stablemimic_reference.h"

#include "g1_config_utils.h"
#include "g1_control_options.h"
#include "g1_joint_mapping.h"
#include "g1_loop_scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

namespace {
using namespace g1_stablemimic;

template<class Read>
void ReadStep(Read read, State& state, Reference& reference) {
    read("q", state.position);
    read("dq", state.velocity);
    read("pelvis_rotation", state.pelvis_rotation);
    read("pelvis_omega_world", state.pelvis_angular_velocity_world);
    read("pelvis_velocity_world", state.pelvis_linear_velocity_world);
    read("torso_position_world", state.torso_position_world);
    read("torso_rotation", state.torso_rotation);
    read("reference_q", reference.position);
    read("reference_dq", reference.velocity);
    read("reference_torso_position_world", reference.torso_position_world);
    read("reference_torso_rotation", reference.torso_rotation);
}

template<class Value, class Expected>
double CheckValues(const Value& actual, const Expected& expected, const std::string& label) {
    constexpr double tolerance = 3e-4;
    double worst = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double error = std::abs(static_cast<double>(actual[index]) - expected[index]);
        if (!std::isfinite(error) || error > tolerance) {
            throw std::runtime_error(label + " mismatch at index " + std::to_string(index) +
                                     ", absolute error " + std::to_string(error));
        }
        worst = std::max(worst, error);
    }
    return worst;
}

int CheckPolicy(Policy& policy, const std::string& fixture_path) {
    const auto fixture = YAML::LoadFile(fixture_path);
    const auto sequences = fixture["sequences"];
    if (fixture["format"].as<std::string>() != "g1-stablemimic-check-v1" ||
        !sequences.IsSequence() || !sequences.size()) {
        throw std::runtime_error("Expected a StableMimic parity fixture");
    }
    const auto drive = fixture["drive_dds"];
    const auto& actual = policy.Drive();
    CheckValues(actual.kp, g1_config::ReadArray<double, kJointCount>(drive["stiffness"]), "DDS Kp");
    CheckValues(actual.kd, g1_config::ReadArray<double, kJointCount>(drive["damping"]), "DDS Kd");
    CheckValues(actual.armature, g1_config::ReadArray<double, kJointCount>(drive["armature"]), "DDS armature");
    CheckValues(actual.effort_limit, g1_config::ReadArray<double, kJointCount>(drive["joint_effort_limit"]), "DDS effort");
    CheckValues(actual.velocity_limit, g1_config::ReadArray<double, kJointCount>(drive["joint_velocity_limit"]), "DDS velocity limit");

    for (const auto& sequence : sequences) {
        const auto name = sequence["name"].as<std::string>();
        const auto steps = sequence["steps"];
        if (!steps.IsSequence() || steps.size() <= kHistoryFrames) {
            throw std::runtime_error(name + " must exercise a full history window and its rollover");
        }
        policy.Reset();
        double observation_error = 0, action_error = 0, gate_error = 0, target_error = 0;
        for (std::size_t index = 0; index < steps.size(); ++index) {
            const auto step = steps[index];
            const auto label = name + " step " + std::to_string(index);
            State state;
            Reference reference;
            ReadStep([&](const char* key, auto& values) {
                using Values = std::decay_t<decltype(values)>;
                values = g1_config::ReadArray<double, std::tuple_size_v<Values>>(step[key]);
            }, state, reference);
            Observations inputs;
            const auto result = policy.Step(state, reference, &inputs);

            // Reference frames are stored once. Reconstruct each expected window
            // from its source indices, including bootstrapping the first frame.
            Observations expected;
            for (int slot = 0; slot < kHistoryFrames; ++slot) {
                const auto distance = static_cast<std::size_t>(kHistoryFrames - 1 - slot);
                const auto history = steps[index > distance ? index - distance : 0];
                const auto actor_frame = g1_config::ReadArray<float, kPolicyFrameSize>(history["policy_frame"]);
                const auto gate_frame = g1_config::ReadArray<float, kGateFrameSize>(history["gate_frame"]);
                std::copy(actor_frame.begin(), actor_frame.end(), expected.policy.begin() + slot * kPolicyFrameSize);
                std::copy(gate_frame.begin(), gate_frame.end(), expected.gate.begin() + slot * kGateFrameSize);
            }
            observation_error = std::max({observation_error,
                CheckValues(inputs.policy, expected.policy, label + " policy history"),
                CheckValues(inputs.gate, expected.gate, label + " gate history")});
            action_error = std::max(action_error, CheckValues(result.action,
                g1_config::ReadArray<float, kJointCount>(step["action"]), label + " action"));
            gate_error = std::max(gate_error, CheckValues(result.gate_weights,
                g1_config::ReadArray<float, 2>(step["gate_weights"]), label + " gate weights"));
            target_error = std::max(target_error, CheckValues(result.target,
                g1_config::ReadArray<double, kJointCount>(step["target_dds"]), label + " DDS target"));
        }
        std::cout << name << ": " << steps.size() << " steps, max abs error"
                  << " obs=" << observation_error << " action=" << action_error << " gate=" << gate_error
                  << " target=" << target_error << " rad\n";
    }
    std::cout << "StableMimic check passed. Policy period " << kPolicyPeriod.count() << " ms.\n";
    return 0;
}

template<class Values>
void WriteValues(const Values& values) {
    for (auto value : values) std::cout << value << ' ';
}

int RunStream(Policy& policy) {
    std::cout << "g1-stablemimic-v1 " << kPolicyPeriod.count();
    for (const auto& name : g1_joints::kDdsNames) std::cout << ' ' << name;
    std::cout << std::endl << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command, extra;
        input >> command;
        if (command == "reset") {
            if (input >> extra) throw std::runtime_error("Unexpected data after StableMimic reset");
            policy.Reset();
            std::cout << "ok" << std::endl;
            continue;
        }
        if (command != "step" && command != "inspect") throw std::runtime_error("Unknown StableMimic stream command");
        State state;
        Reference reference;
        ReadStep([&](const char* key, auto& values) {
            for (auto& value : values) {
                if (!(input >> value) || !std::isfinite(value)) {
                    throw std::runtime_error(std::string("Invalid StableMimic stream field: ") + key);
                }
            }
        }, state, reference);
        if (input >> extra) throw std::runtime_error("Unexpected StableMimic stream data");
        const bool inspect = command == "inspect";
        Observations observations;
        const auto result = policy.Step(state, reference, inspect ? &observations : nullptr);
        WriteValues(result.action);
        WriteValues(result.gate_weights);
        WriteValues(result.target);
        if (inspect) {
            WriteValues(observations.policy);
            WriteValues(observations.gate);
        }
        std::cout << std::endl;
    }
    return 0;
}

int RunHardwareStream(Policy& policy, const HardwareSettings& settings, Playback& playback) {
    constexpr auto control_period = g1_control::kControlPeriod;
    static_assert(kPolicyPeriod % control_period == std::chrono::milliseconds::zero());
    constexpr auto steps_per_inference = kPolicyPeriod / control_period;
    std::cout << "g1-stablemimic-hardware-v1 " << kPolicyPeriod.count() << ' ' << control_period.count();
    for (const auto& name : g1_joints::kDdsNames) std::cout << ' ' << name;
    std::cout << std::endl << std::setprecision(std::numeric_limits<double>::max_digits10);

    Result result;
    Playback::Sample sample{};
    std::optional<Actuation> telemetry;
    std::uint64_t tick = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command, extra;
        input >> command;
        if (command == "reset") {
            if (input >> extra) throw std::runtime_error("Unexpected data after StableMimic reset");
            policy.Reset();
            playback.Reset();
            telemetry.reset();
            tick = 0;
            std::cout << "ok" << std::endl;
            continue;
        }
        if (command == "telemetry") {
            if (input >> extra) throw std::runtime_error("Unexpected data after StableMimic telemetry request");
            if (!telemetry) {
                std::cout << "none" << std::endl;
                continue;
            }
            WriteValues(telemetry->position);
            WriteValues(telemetry->kp);
            WriteValues(telemetry->kd);
            std::cout << std::endl;
            continue;
        }
        if (command != "check" && command != "damping" && command != "track")
            throw std::runtime_error("Unknown StableMimic hardware stream command");
        State state;
        Vector3 pelvis_position;
        auto read = [&input](auto& values) {
            for (auto& value : values)
                if (!(input >> value) || !std::isfinite(value))
                    throw std::runtime_error("Invalid StableMimic hardware simulation state");
        };
        try {
            read(state.position);
            read(state.velocity);
            read(pelvis_position);
            read(state.pelvis_rotation);
            read(state.pelvis_angular_velocity_world);
            read(state.pelvis_linear_velocity_world);
            if (input >> extra) throw std::runtime_error("Unexpected StableMimic hardware stream data");
            CheckMeasuredPositions(settings, state.position);
            if (command == "check") {
                std::cout << (Resting(settings, state) ? "ready" : "unready") << std::endl;
                continue;
            }
            const Joints* target = nullptr;
            float weight = 0;
            const bool infer = command == "track" && tick % steps_per_inference == 0;
            if (command == "track") {
                if (infer) {
                    SetTorsoPose(state, pelvis_position);
                    sample = playback.Step(state, pelvis_position);
                    result = policy.Step(state, sample.command);
                }
                target = &result.target;
                weight = RampWeight(settings, control_period.count() * 0.001 * tick);
                ++tick;
            }
            const auto actuation = ComputeActuation(settings, policy.Drive(), state, target, weight);
            telemetry = actuation;
            std::cout << command << ' ';
            if (target) {
                std::cout << sample.frame << ' ' << (infer && sample.realigned) << ' ' << sample.holding << ' ';
                WriteValues(result.gate_weights);
            }
            WriteValues(actuation.torque);
            std::cout << std::endl;
        } catch (const std::runtime_error& error) {
            telemetry.reset();
            std::cout << "fault " << error.what() << std::endl;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "--help";
        if (mode == "--help" || mode == "help") {
            std::cout << "Usage: g1_stablemimic check [--config JSON] [--model ONNX] [--fixture JSON]\n"
                      << "       g1_stablemimic sim-stream [--config JSON] [--model ONNX]\n"
                      << "       g1_stablemimic sim-hardware-stream [--config JSON] [--model ONNX]"
                         " [--motion REFERENCE.txt] [--start-frame INDEX]\n";
            return 0;
        }
        if (mode != "check" && mode != "sim-stream" && mode != "sim-hardware-stream")
            throw std::invalid_argument("Unknown StableMimic subcommand: " + mode);
        std::string config = G1_STABLEMIMIC_DEFAULT_CONFIG;
        std::optional<std::string> model, fixture;
        std::string motion;
        int start_frame = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--config") config = g1_control::OptionValue(argc, argv, i, option);
            else if (option == "--model") model = g1_control::OptionValue(argc, argv, i, option);
            else if (option == "--fixture" && mode == "check") fixture = g1_control::OptionValue(argc, argv, i, option);
            else if (option == "--motion" && mode == "sim-hardware-stream")
                motion = g1_control::OptionValue(argc, argv, i, option);
            else if (option == "--start-frame" && mode == "sim-hardware-stream")
                start_frame = g1_control::ParseInteger(g1_control::OptionValue(argc, argv, i, option));
            else throw std::invalid_argument("Unknown option: " + option);
        }
        const auto settings = YAML::LoadFile(config);
        if (!model) model = g1_config::ResolvePath(config, settings["policy_path"].as<std::string>());
        Policy policy(*model);
        if (mode == "sim-stream") return RunStream(policy);
        if (mode == "sim-hardware-stream") {
            if (start_frame < 0) throw std::invalid_argument("Start frame must be nonnegative");
            const HardwareSettings hardware(settings);
            Playback playback(LoadStandingReference(g1_config::ResolvePath(config, hardware.standing_reference)),
                              motion.empty() ? std::vector<ReferenceFrame>{} : LoadReference(motion),
                              static_cast<std::size_t>(start_frame));
            return RunHardwareStream(policy, hardware, playback);
        }
        if (!fixture) fixture = g1_config::ResolvePath(config, settings["check_path"].as<std::string>());
        return CheckPolicy(policy, *fixture);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
