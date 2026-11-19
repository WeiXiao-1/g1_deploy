#include "g1_getup_mimic_control.h"

#include "g1_config_utils.h"
#include "g1_control_options.h"
#include "g1_joint_mapping.h"
#include "g1_loop_scheduler.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace g1_getup_mimic;

constexpr float kTolerance = 3e-4f;

void PrintUsage(const char* program) {
    std::cout << "Usage: " << program << " check [--config JSON] [--expert NAME]\n"
              << "Check GetUpMimic model and joint targets offline.\n"
              << "For MuJoCo, use apps/simulation/sim.py --model_test [--actuation drive_pd|host_torque].\n";
}

float CheckJoints(const Joints& actual, const YAML::Node& expected_node, const std::string& label) {
    const auto expected = g1_config::ReadArray<float, kJointCount>(expected_node);
    float max_error = 0;
    for (int joint = 0; joint < kJointCount; ++joint) {
        const float error = std::abs(actual[joint] - expected[joint]);
        if (!std::isfinite(error) || error > kTolerance) {
            throw std::runtime_error(label + " mismatch at index " + std::to_string(joint) +
                                     ", absolute error " + std::to_string(error));
        }
        max_error = std::max(max_error, error);
    }
    return max_error;
}

int CheckPolicy(const PolicyConfig& config) {
    const auto check_path = g1_config::ResolvePath(config.path, config.model["check_path"].as<std::string>());
    const auto checks = YAML::LoadFile(check_path);
    const auto parity = checks["parity"];
    const auto sequences = checks["sequences"];
    if (!parity.IsSequence() || parity.size() == 0 || !sequences.IsSequence() || sequences.size() == 0) {
        throw std::runtime_error("GetUpMimic check requires model parity samples and state sequences");
    }

    Policy policy(config);
    float parity_error = 0;
    for (std::size_t sample = 0; sample < parity.size(); ++sample) {
        auto observation = parity[sample]["obs"].as<std::vector<float>>();
        const auto error = CheckJoints(policy.Predict(observation.data(), observation.size()), parity[sample]["actions"],
                                       "ONNX parity sample " + std::to_string(sample));
        parity_error = std::max(parity_error, error);
    }
    std::cout << "GetUpMimic ONNX parity: " << parity.size() << " samples, max action error " << parity_error << '\n';

    for (const auto& sequence : sequences) {
        const auto label = sequence["name"].as<std::string>();
        const auto steps = sequence["steps"];
        const std::size_t minimum_steps = config.observation_size == kObservationSize ? kHistoryFrames + 1 : 2;
        if (!steps.IsSequence() || steps.size() < minimum_steps) {
            throw std::runtime_error(label + " needs at least " + std::to_string(minimum_steps) + " consecutive steps");
        }
        policy.Reset();
        float target_error = 0;
        for (std::size_t index = 0; index < steps.size(); ++index) {
            const auto step = steps[index];
            State state;
            state.position = g1_config::ReadArray<float, kJointCount>(step["q"]);
            state.velocity = g1_config::ReadArray<float, kJointCount>(step["dq"]);
            state.pelvis_quaternion = g1_config::ReadArray<float, 4>(step["pelvis_quaternion"]);
            state.pelvis_angular_velocity = g1_config::ReadArray<float, 3>(step["pelvis_angular_velocity"]);
            const auto targets = policy.Step(state);
            if (index == 0) {
                CheckJoints(targets.kp, checks["kp"], "DDS Kp");
                CheckJoints(targets.kd, checks["kd"], "DDS Kd");
            }
            const auto error = CheckJoints(targets.position, step["target"], label + " step " + std::to_string(index));
            target_error = std::max(target_error, error);
        }
        std::cout << label << ": " << steps.size() << " steps, max DDS target error " << target_error << " rad\n";
    }
    std::cout << "GetUpMimic check passed. Policy period " << kPolicyPeriod.count() << " ms.\n";
    return 0;
}

template<class Values>
void ReadStateValues(std::istream& input, Values& values) {
    for (float& value : values) {
        if (!(input >> value)) {
            throw std::runtime_error("Incomplete or invalid GetUpMimic simulation state");
        }
    }
}

State ReadSimulationState(std::istream& input) {
    State state;
    ReadStateValues(input, state.position);
    ReadStateValues(input, state.velocity);
    ReadStateValues(input, state.pelvis_quaternion);
    ReadStateValues(input, state.pelvis_angular_velocity);
    return state;
}

void RequireEndOfCommand(std::istream& input) {
    std::string extra;
    if (input >> extra) {
        throw std::runtime_error("Unexpected values in GetUpMimic simulation command");
    }
}

void WriteJointNames() {
    for (const auto& joint : g1_joints::kDdsNames) {
        std::cout << ' ' << joint;
    }
    std::cout << std::endl << std::setprecision(std::numeric_limits<float>::max_digits10);
}

void WriteJointValues(const Joints& values) {
    for (float value : values) std::cout << ' ' << value;
}

void WriteTargets(const MotorTargets& targets) {
    WriteJointValues(targets.position);
    WriteJointValues(targets.kp);
    WriteJointValues(targets.kd);
}

// The simulator owns the clock. Names in the handshake specify the wire order.
int RunSimulationStream(const PolicyConfig& config) {
    Policy policy(config);
    std::cout << "g1-getup-mimic-v1 " << kPolicyPeriod.count();
    WriteJointNames();

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        if (command == "reset") {
            RequireEndOfCommand(input);
            policy.Reset();
            std::cout << "ok" << std::endl;
            continue;
        }
        if (command != "step") throw std::runtime_error("Unknown GetUpMimic simulation command");
        const auto state = ReadSimulationState(input);
        RequireEndOfCommand(input);
        WriteTargets(policy.Step(state));
        std::cout << std::endl;
    }
    return 0;
}

int RunHardwareSimulationStream(const PolicyConfig& config, std::optional<ActuationMode> actuation_override) {
    const HardwareSettings settings(config.root, actuation_override);
    const bool drive_pd = settings.actuation == ActuationMode::DrivePd;
    Policy policy(config);
    constexpr auto control_period = g1_control::kControlPeriod;
    static_assert(kPolicyPeriod % control_period == std::chrono::milliseconds::zero());
    constexpr auto steps_per_inference = kPolicyPeriod / control_period;
    std::cout << (drive_pd ? "g1-getup-mimic-drive-pd-v1 " : "g1-getup-mimic-hardware-v1 ")
              << kPolicyPeriod.count() << ' ' << control_period.count() << ' ' << settings.timeout_seconds
              << ' ' << settings.limits.stop_kd << ' ' << settings.limits.damping_seconds;
    if (drive_pd) WriteJointValues(settings.effort);
    WriteJointNames();

    MotorTargets targets{};
    std::optional<MotorTargets> telemetry;
    RecoveryProgress progress;
    int tick = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        if (command == "reset") {
            RequireEndOfCommand(input);
            policy.Reset();
            progress.Reset();
            telemetry.reset();
            tick = 0;
            std::cout << "ok" << std::endl;
            continue;
        }
        if (command == "telemetry") {
            RequireEndOfCommand(input);
            if (!telemetry) {
                std::cout << "none" << std::endl;
                continue;
            }
            WriteTargets(*telemetry);
            std::cout << std::endl;
            continue;
        }
        if (command != "check" && command != "damping" && command != "getup") {
            throw std::runtime_error("Unknown GetUpMimic hardware simulation command");
        }
        const auto state = ReadSimulationState(input);
        RequireEndOfCommand(input);
        try {
            CheckMeasuredPositions(settings, state);
            if (command == "check") {
                ProjectedGravity(state.pelvis_quaternion);  // Validate the simulated IMU quaternion.
                std::cout << (ReadyToGetUp(settings, state) ? "ready" : "unready") << std::endl;
                continue;
            }
            const char* phase = "damping";
            const MotorTargets* active_targets = nullptr;
            float weight = 0.f;
            if (command == "getup") {
                const double elapsed = std::chrono::duration<double>(control_period * tick).count();
                const auto status = progress.Update(settings, state, elapsed);
                if (status == RecoveryProgress::Status::TimedOut) {
                    phase = "timeout";
                } else {
                    if (tick % steps_per_inference == 0) targets = policy.Step(state);
                    ++tick;
                    active_targets = &targets;
                    weight = RampWeight(settings, elapsed);
                    phase = status == RecoveryProgress::Status::Standing ? "standing" : "rising";
                }
            }
            const auto actuation = ComputeActuation(settings, state, active_targets, weight);
            telemetry = actuation.applied;
            std::cout << phase;
            if (drive_pd) WriteTargets(actuation.applied);
            else WriteJointValues(actuation.torque);
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
        if (argc == 1 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
            PrintUsage(argv[0]);
            return 0;
        }
        const std::string mode = argv[1];
        if (mode != "check" && mode != "sim-stream" && mode != "sim-hardware-stream") {
            throw std::invalid_argument("Expected GetUpMimic subcommand check");
        }
        std::string config_path = G1_GETUP_MIMIC_DEFAULT_CONFIG;
        std::string expert;
        std::optional<ActuationMode> actuation;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                PrintUsage(argv[0]);
                return 0;
            }
            if (option == "--config") {
                config_path = g1_control::OptionValue(argc, argv, i, option);
            } else if (option == "--expert") {
                expert = g1_control::OptionValue(argc, argv, i, option);
                if (expert.empty()) throw std::invalid_argument("Expected a model_test expert name");
            } else if (option == "--actuation" && mode == "sim-hardware-stream") {
                actuation = ParseActuationMode(g1_control::OptionValue(argc, argv, i, option));
            } else {
                throw std::invalid_argument("Unknown option: " + option);
            }
        }
        const PolicyConfig config(config_path, expert);
        if (mode == "check") {
            if (!expert.empty()) std::cout << "model_test expert " << expert << '\n';
            return CheckPolicy(config);
        }
        if (mode == "sim-hardware-stream") return RunHardwareSimulationStream(config, actuation);
        return RunSimulationStream(config);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
