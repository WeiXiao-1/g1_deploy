#include "g1_stablemimic_reference.h"

#include "g1_async_log.h"
#include "g1_control_options.h"
#include "g1_control_runtime.h"
#include "g1_control_signals.h"
#include "g1_control_targets.h"
#include "g1_dds_state_cache.h"
#include "g1_hardware.h"
#include "g1_joint_mapping.h"
#include "g1_keyboard.h"
#include "g1_loop_scheduler.h"
#include "g1_monitor_publisher.h"
#include "g1_motion_service.h"
#include "g1_motor_channel.h"
#include "g1_policy_worker.h"
#include "g1_unitree_remote.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unitree/idl/go2/SportModeState_.hpp>

namespace {
using namespace g1_stablemimic;
using LowState = unitree_hg::msg::dds_::LowState_;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using BaseState = unitree_go::msg::dds_::SportModeState_;
using Clock = std::chrono::steady_clock;
using Receiver = g1_control::DdsStateCache<LowState>;
using BaseReceiver = g1_control::DdsStateCache<BaseState>;
using Snapshot = g1_control::StateCache<LowState>::Snapshot;

double PositiveSetting(const YAML::Node& node, const char* name, double maximum) {
    const auto value = node[name].as<double>();
    if (!std::isfinite(value) || value <= 0 || value > maximum)
        throw std::runtime_error(std::string("Invalid StableMimic hardware setting: ") + name);
    return value;
}

struct Settings {
    g1_hardware::LimitsConfig limits;
    std::string base_topic, standing_reference;
    double estimate_timeout, max_skew, ramp_seconds, position_tolerance;
    double entry_joint_speed, entry_pelvis_speed, entry_gyro;
    Joints effort, velocity;

    explicit Settings(const YAML::Node& root) : limits(root, false) {
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
};

template<std::size_t N, class T>
std::array<double, N> AsDoubles(const std::array<T, N>& array) {
    std::array<double, N> result{};
    std::copy(array.begin(), array.end(), result.begin());
    return result;
}

double Norm(const Vector3& a) { return std::hypot(a[0], a[1], a[2]); }

void CheckMotorState(const Settings& settings, const LowState& state) {
    g1_hardware::CheckMotorHealth(state, settings.limits.temperature);
    for (int joint = 0; joint < kJointCount; ++joint) {
        const double q = state.motor_state()[joint].q();
        if (q < settings.limits.lower[joint] - settings.position_tolerance ||
            q > settings.limits.upper[joint] + settings.position_tolerance)
            throw std::runtime_error("Measured joint outside limits: " + g1_joints::kDdsNames[joint] +
                                     " at " + std::to_string(q) + " rad");
    }
}

struct ObservedState {
    State state;
    Vector3 pelvis_position;
    std::uint64_t sequence;
};

ObservedState AssembleState(const LowState& low, const BaseState& estimate, std::uint64_t sequence) {
    ObservedState result{{}, AsDoubles(estimate.position()), sequence};
    auto& state = result.state;
    for (int joint = 0; joint < kJointCount; ++joint) {
        state.position[joint] = low.motor_state()[joint].q();
        state.velocity[joint] = low.motor_state()[joint].dq();
    }
    state.pelvis_rotation = FromQuaternion(AsDoubles(estimate.imu_state().quaternion()));
    const auto low_rotation = FromQuaternion(AsDoubles(low.imu_state().quaternion()));
    // Gravity direction is independent of the odometer's world yaw.
    double up_agreement = 0;
    for (int i = 6; i < 9; ++i) up_agreement += low_rotation[i] * state.pelvis_rotation[i];
    if (up_agreement < std::cos(20.0 * std::acos(-1.) / 180.))
        throw std::runtime_error("Base estimate and LowState disagree on gravity direction");

    state.pelvis_angular_velocity_world = Rotate(state.pelvis_rotation, AsDoubles(low.imu_state().gyroscope()));
    // Translate the velocity of the pelvis link origin to the pelvis link COM.
    const auto com_offset = Rotate(state.pelvis_rotation, {0, 0, -.07605});
    const auto com_correction = Cross(state.pelvis_angular_velocity_world, com_offset);
    for (int i = 0; i < 3; ++i)
        state.pelvis_linear_velocity_world[i] = estimate.velocity()[i] + com_correction[i];
    SetTorsoPose(state, result.pelvis_position);
    return result;
}

void StartBaseReceiver(BaseReceiver& receiver, const std::atomic<bool>& armed) {
    struct Previous {
        std::int64_t stamp = 0;
        Vector3 position{};
        Rotation rotation{};
    };
    receiver.Start([&armed, previous = std::make_shared<Previous>()](const BaseState& message) {
        const auto& stamp = message.stamp();
        const auto time = std::int64_t(stamp.sec()) * 1000000000LL + stamp.nanosec();
        if (stamp.nanosec() >= 1000000000u || time <= previous->stamp)
            throw std::runtime_error("Base estimate timestamp must strictly increase");
        if (message.error_code() != 0 ||
            !std::all_of(message.position().begin(), message.position().end(), [](float x) { return std::isfinite(x); }) ||
            message.position()[2] < -.25f || message.position()[2] > 1.5f || !(Norm(AsDoubles(message.velocity())) <= 8))
            throw std::runtime_error("Invalid pelvis position, velocity or estimator status");
        const auto rotation = FromQuaternion(AsDoubles(message.imu_state().quaternion()));
        const auto position = AsDoubles(message.position());
        if (armed && previous->stamp != 0) {
            const double dt = double(time - previous->stamp) * 1e-9;
            const Vector3 delta{position[0] - previous->position[0], position[1] - previous->position[1],
                                position[2] - previous->position[2]};
            double rotation_trace = 0;
            for (std::size_t i = 0; i < rotation.size(); ++i) rotation_trace += rotation[i] * previous->rotation[i];
            const double angle = std::acos(std::clamp((rotation_trace - 1.) / 2., -1., 1.));
            if (Norm(delta) > .10 + 8.*dt || angle > .25 + 15.*dt)
                throw std::runtime_error("Discontinuous pelvis odometry");
        }
        *previous = {time, position, rotation};
    }, [&armed] { return armed.load(); });
}

void StartLowReceiver(Receiver& receiver, const Settings& settings, bool read_only,
                      const std::atomic<bool>& armed, std::atomic<bool>& remote_stop) {
    receiver.Start([&, read_only](const LowState& state) {
        if (armed || read_only) CheckMotorState(settings, state);
        else g1_hardware::CheckTelemetryState(state);
        if (!read_only && (state.wireless_remote()[2] & (1u << 3))) remote_stop = true;
    }, [&, read_only] { return armed || read_only; });
}

Snapshot ReadLow(const Receiver& receiver, const Settings& settings) {
    const auto low = receiver.Read();
    if (!low.valid || low.AgeMs(Clock::now()) > settings.limits.timeout_ms)
        throw std::runtime_error("LowState stale or invalid. " + low.error);
    return low;
}

ObservedState ReadPair(const Snapshot& low, const BaseReceiver& receiver, const Settings& settings,
                       Clock::time_point after = {}, std::chrono::system_clock::time_point sampled_after = {}) {
    const auto base = receiver.Read();
    const auto& stamp = base.value.stamp();
    const auto source = std::chrono::seconds(stamp.sec()) + std::chrono::nanoseconds(stamp.nanosec());
    const double source_age = std::chrono::duration<double, std::milli>(
        std::chrono::system_clock::now().time_since_epoch() - source).count();
    const auto skew = std::chrono::duration<double, std::milli>(base.received_at - low.received_at).count();
    if (!base.valid || base.received_at <= after || source <= sampled_after.time_since_epoch() ||
        base.AgeMs(Clock::now()) > settings.estimate_timeout ||
        source_age < -20. || source_age > settings.estimate_timeout || std::abs(skew) > settings.max_skew)
        throw std::runtime_error("Base estimate missing, stale or out of sync on " + settings.base_topic +
                                 ". " + base.error);
    const auto sequence = static_cast<std::uint64_t>(low.received_at.time_since_epoch().count());
    return AssembleState(low.value, base.value, sequence);
}

bool Resting(const Settings& settings, const ObservedState& observed) {
    const auto& state = observed.state;
    return Norm(state.pelvis_angular_velocity_world) <= settings.entry_gyro &&
           Norm(state.pelvis_linear_velocity_world) <= settings.entry_pelvis_speed &&
           std::all_of(state.velocity.begin(), state.velocity.end(),
                       [&](double dq) { return std::abs(dq) <= settings.entry_joint_speed; });
}

struct Commands { LowCmd motors, monitor; };

Commands BuildCommands(const Settings& settings, const DriveParameters& drive, const LowState& state,
                       const Joints* target, double weight) {
    Commands commands{};
    commands.motors.mode_machine() = commands.monitor.mode_machine() = state.mode_machine();
    for (int i = 0; i < kJointCount; ++i) {
        const double q = state.motor_state()[i].q(), dq = state.motor_state()[i].dq();
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

        auto& motor = commands.motors.motor_cmd()[i];
        motor.mode() = 1;
        motor.q() = static_cast<float>(goal);
        motor.tau() = static_cast<float>(std::clamp(torque, -effort, effort));
        auto& monitor = commands.monitor.motor_cmd()[i];
        monitor = motor;
        monitor.kp() = static_cast<float>(kp);
        monitor.kd() = static_cast<float>(kd);
        monitor.tau() = 0;
    }
    g1_hardware::CheckCommand(commands.motors);
    commands.motors.crc() = g1_hardware::Crc(commands.motors);
    return commands;
}

struct Inference {
    Result output;
    std::size_t frame;
    bool realigned, holding;
    double inference_ms;
};

class Controller {
    using Worker = g1_inference::PolicyWorker<ObservedState, Inference>;

public:
    Controller(const Settings& settings, const Receiver& low, const BaseReceiver& base, Policy& policy,
               Playback& playback, std::atomic<bool>& armed, const std::atomic<bool>& remote_stop)
        : settings_(settings), low_(low), base_(base), drive_(policy.Drive()), armed_(armed),
          remote_stop_(remote_stop), worker_(
              [&policy, &playback](const ObservedState& input, const auto&) {
                  const auto start = Clock::now();
                  const auto sample = playback.Step(input.state, input.pelvis_position);
                  auto result = policy.Step(input.state, sample.command);
                  const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                  return Inference{result, sample.frame, sample.realigned, sample.holding, elapsed};
              }, [&policy, &playback] { policy.Reset(); playback.Reset(); }, true, kPolicyPeriod) {}

    int Run() {
        telemetry_.Start();
        telemetry_.PublishEnable(false);
        std::cout << "StableMimic DRY RUN. P/X previews, 9/Start takes over, Space/Select exits.\n" << std::flush;
        g1_control::LoopScheduler<> cycle;
        bool failed = false;
        try {
            while (!Stopped()) {
                auto low = ReadLow(low_, settings_);
                remote_.Update(low.value.wireless_remote());
                const char action = g1_control::SelectAction(keyboard_.Read(false), remote_.state.Action());
                if (action == ' ') break;
                if (action == '9' && !armed_) {
                    TakeOver(low);
                    cycle.Sleep();
                    continue;
                }
                std::optional<ObservedState> observed;
                if (armed_ || running_ || action == 'p' || action == 'P') {
                    try {
                        observed = ReadPair(low, base_, settings_);
                    } catch (const std::exception& error) {
                        if (armed_) throw;
                        HaltPolicy();
                        log_.Line() << "Dry run: " << error.what() << '\n';
                    }
                }
                if (armed_ && monitor_->Conflict()) throw std::runtime_error("Competing LowCmd publisher");
                if ((action == 'p' || action == 'P') && !running_ && observed) Begin(*observed);
                Update(low, observed);
                cycle.Sleep();
            }
        } catch (const std::exception& error) {
            log_.Line() << "StableMimic stopped: " << error.what() << '\n';
            failed = true;
        }
        HaltPolicy();
        try {
            if (armed_) ReleaseOutput(failed && !Stopped());
            telemetry_.PublishEnable(false);
        } catch (const std::exception& error) {
            log_.Line() << "StableMimic stop output failed: " << error.what() << '\n';
            failed = true;
        }
        return failed ? 1 : 0;
    }

private:
    bool Stopped() const { return g1_control::StopRequested() || remote_stop_; }

    bool EntryReady(const ObservedState& state) const {
        return g1_control::InputReady(&remote_.state) && Resting(settings_, state);
    }

    void TakeOver(const Snapshot& low) {
        HaltPolicy();
        try {
            CheckMotorState(settings_, low.value);
            if (!EntryReady(ReadPair(low, base_, settings_)))
                throw std::runtime_error("Center the sticks and wait for the robot to settle");
        } catch (const std::exception& error) {
            log_.Line() << "Takeover rejected: " << error.what() << '\n';
            return;
        }
        if (Stopped()) return;
        monitor_ = std::make_unique<g1_control::CommandMonitor>();
        publisher_ = std::make_unique<g1_control::MotorPublisher>("rt/lowcmd");
        publisher_->InitChannel();
        g1_control::ReleaseMotionService();
        const auto released = Clock::now();
        const auto released_wall_time = std::chrono::system_clock::now();
        monitor_->Arm(*publisher_);

        // Require a new pelvis estimate and new motor feedback after service release.
        const auto end = released + std::chrono::duration<double, std::milli>(settings_.estimate_timeout);
        std::string reason = "No new LowState after release";
        while (Clock::now() < end && !Stopped()) {
            for (char key; (key = keyboard_.Read(false));) {
                if (key == ' ') {
                    g1_control::RequestStop(0);
                    return;
                }
            }
            if (monitor_->Conflict()) throw std::runtime_error("Competing LowCmd publisher");
            const auto fresh = low_.Read();
            if (fresh.valid && fresh.received_at > released && fresh.AgeMs(Clock::now()) <= settings_.limits.timeout_ms) {
                try {
                    CheckMotorState(settings_, fresh.value);
                    const auto observed = ReadPair(fresh, base_, settings_, released, released_wall_time);
                    if (!Resting(settings_, observed))
                        throw std::runtime_error("Robot moved during takeover");
                    remote_ = g1_unitree_remote::Input{};
                    remote_.Update(fresh.value.wireless_remote());  // Discard queued X.
                    if (Stopped()) return;
                    if (!EntryReady(observed)) throw std::runtime_error("Center the sticks after takeover");
                    mode_machine_ = fresh.value.mode_machine();
                    armed_ = true;
                    log_.Line() << "Control enabled. Damping only. Fresh P/X starts StableMimic.\n";
                    return;
                } catch (const std::exception& error) {
                    reason = error.what();
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (!Stopped()) throw std::runtime_error("Takeover incomplete after motion service release: " + reason);
    }

    void Begin(const ObservedState& state) {
        if (armed_ && !EntryReady(state)) {
            log_.Line() << "StableMimic start requires centered sticks and a resting robot.\n";
            return;
        }
        worker_.Begin(true);
        running_ = true;
        ramp_started_.reset();
        last_reported_ = {};
        next_report_ = Clock::now() + std::chrono::seconds(1);
        peak_ms_ = 0;
        holding_ = false;
        log_.Line() << (armed_ ? "StableMimic policy ON.\n" : "StableMimic preview ON.\n");
    }

    void HaltPolicy() {
        worker_.Pause();
        running_ = false;
    }

    void Update(const Snapshot& low, const std::optional<ObservedState>& observed) {
        std::optional<Worker::Result> latest;
        if (running_) {
            worker_.Submit(*observed, {});
            latest = worker_.Latest();
        }
        telemetry_.PublishEnable(armed_);
        if (!armed_ && !latest) return;
        const auto now = Clock::now();
        double weight = 0;
        if (latest) {
            if (!ramp_started_) ramp_started_ = now;
            const double elapsed = std::chrono::duration<double>(now - *ramp_started_).count();
            weight = g1_control::InterpolateReset(0.f, 1.f, g1_control::ResetProgress(elapsed, settings_.ramp_seconds));
            Report(*latest, now);
        }
        const auto commands = BuildCommands(settings_, drive_, low.value,
                                             latest ? &latest->target.output.target : nullptr, weight);
        if (armed_) {
            if (Stopped()) return;
            if (monitor_->Conflict()) throw std::runtime_error("Competing LowCmd publisher");
            if (!publisher_->Write(commands.motors)) throw std::runtime_error("LowCmd publication failed");
            telemetry_.PublishTarget(commands.monitor);
        } else {
            telemetry_.PublishPreview(commands.monitor);
        }
    }

    void Report(const Worker::Result& latest, Clock::time_point now) {
        if (latest.completed_at == last_reported_) return;
        last_reported_ = latest.completed_at;
        const auto& v = latest.target;
        if (v.realigned) log_.Line() << "Recovery XY realigned at reference frame " << v.frame << ".\n";
        if (v.holding && !holding_) log_.Line() << "Using standing reference.\n";
        holding_ = v.holding;
        peak_ms_ = std::max(peak_ms_, v.inference_ms);
        if (now >= next_report_) {
            log_.Line() << "Frame " << v.frame << " | gate (tracking, recovery): " << v.output.gate_weights[0]
                        << ", " << v.output.gate_weights[1] << " | peak " << peak_ms_ << " ms / 20 ms.\n";
            peak_ms_ = 0;
            next_report_ = now + std::chrono::seconds(1);
        }
    }

    void ReleaseOutput(bool fault) {
        LowCmd zero{};
        zero.mode_machine() = mode_machine_;
        g1_control::SetDampingTargets(zero, true, 0);
        zero.crc() = g1_hardware::Crc(zero);
        const auto end = Clock::now() + std::chrono::duration<double>(settings_.limits.damping_seconds);
        g1_control::LoopScheduler<> cycle;
        bool written = false;
        while (!monitor_->Conflict() && Clock::now() < end) {
            auto stop = zero;
            if (fault && !Stopped()) {
                try {
                    const auto fresh = ReadLow(low_, settings_);
                    CheckMotorState(settings_, fresh.value);
                    stop = BuildCommands(settings_, drive_, fresh.value, nullptr, 0.).motors;
                } catch (const std::exception&) {
                    // With invalid motor feedback, send zero torque rather than unbounded drive-side PD.
                }
            }
            if (Stopped()) stop = zero;
            written |= publisher_->Write(stop);
            cycle.Sleep();
        }
        publisher_.reset();
        armed_ = false;
        if (!written && !monitor_->Conflict()) throw std::runtime_error("No stop command could be published");
    }

    const Settings& settings_;
    const Receiver& low_;
    const BaseReceiver& base_;
    const DriveParameters& drive_;
    std::atomic<bool>& armed_;
    const std::atomic<bool>& remote_stop_;
    g1_control::AsyncLog log_;
    g1_control::Keyboard keyboard_;
    g1_unitree_remote::Input remote_;
    g1_control::MonitorPublisher telemetry_;
    std::unique_ptr<g1_control::CommandMonitor> monitor_;
    std::unique_ptr<g1_control::MotorPublisher> publisher_;
    Worker worker_;
    Clock::time_point last_reported_{}, next_report_{};
    std::optional<Clock::time_point> ramp_started_;
    double peak_ms_ = 0;
    std::uint8_t mode_machine_ = 0;
    bool running_ = false, holding_ = false;
};

std::optional<std::pair<Snapshot, ObservedState>> WaitForInputs(
    const Receiver& low, const BaseReceiver& base, const Settings& settings) {
    const auto end = Clock::now() + g1_control::kStartupTimeout;
    std::string reason;
    while (!g1_control::StopRequested() && Clock::now() < end) {
        try {
            auto snapshot = ReadLow(low, settings);
            return std::make_pair(snapshot, ReadPair(snapshot, base, settings));
        } catch (const std::exception& error) {
            reason = error.what();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    if (g1_control::StopRequested()) return std::nullopt;
    throw std::runtime_error("StableMimic startup check: " + reason);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: g1_stablemimic_real {read-only|real} INTERFACE [--config JSON]\n"
                         "       real also accepts [--motion REFERENCE.txt] [--start-frame INDEX]\n";
            return 0;
        }
        const std::string mode = argv[1];
        if (argc < 3 || (mode != "read-only" && mode != "real"))
            throw std::invalid_argument("Specify read-only or real and the robot interface");
        g1_control::RuntimeOptions options{true, 0, argv[2], G1_STABLEMIMIC_DEFAULT_CONFIG};
        g1_control::ValidateRuntimeOptions(options);
        const bool read_only = mode == "read-only";
        std::string motion_path;
        int start_frame = 0;
        for (int i = 3; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--config") options.config = g1_control::OptionValue(argc, argv, i, key);
            else if (!read_only && key == "--motion") motion_path = g1_control::OptionValue(argc, argv, i, key);
            else if (!read_only && key == "--start-frame")
                start_frame = g1_control::ParseInteger(g1_control::OptionValue(argc, argv, i, key));
            else throw std::invalid_argument("Unknown option: " + key);
        }
        if (start_frame < 0) throw std::invalid_argument("Start frame must be nonnegative");

        const auto root = YAML::LoadFile(options.config);
        const Settings settings(root);
        g1_control::InstallSignalHandlers();
        unitree::robot::ChannelFactory::Instance()->Init(options.domain, options.interface);
        std::atomic<bool> armed{false}, remote_stop{false};
        Receiver low("rt/lowstate");
        BaseReceiver base(settings.base_topic);
        StartLowReceiver(low, settings, read_only, armed, remote_stop);
        StartBaseReceiver(base, armed);
        const auto initial = WaitForInputs(low, base, settings);
        if (!initial || remote_stop) return 0;
        const auto& state = initial->second;
        if (read_only) {
            const auto& p = state.pelvis_position;
            const auto& v = state.state.pelvis_linear_velocity_world;
            std::cout << "StableMimic read-only OK | pelvis world xyz [" << p[0] << ", " << p[1] << ", " << p[2]
                      << "] m | pelvis COM velocity [" << v[0] << ", " << v[1] << ", " << v[2]
                      << "] m/s | resting: " << (Resting(settings, state) ? "yes" : "no") << '\n';
            return 0;
        }

        const auto standing = LoadReference(g1_config::ResolvePath(options.config, settings.standing_reference));
        if (standing.size() != 1 || !std::all_of(standing.front().command.velocity.begin(),
                standing.front().command.velocity.end(), [](double dq) { return std::abs(dq) <= 1e-6; }))
            throw std::runtime_error("Standing reference must be one frame with zero joint velocities");
        Playback playback(standing.front(), motion_path.empty() ? std::vector<ReferenceFrame>{} :
                          LoadReference(motion_path), static_cast<std::size_t>(start_frame));
        Policy policy(g1_config::ResolvePath(options.config, root["policy_path"].as<std::string>()));
        policy.Step(state.state, playback.Step(state.state, state.pelvis_position).command);  // Warm up without LowCmd.
        return Controller(settings, low, base, policy, playback, armed, remote_stop).Run();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
