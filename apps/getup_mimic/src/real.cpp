#include "g1_getup_mimic_control.h"
#include "g1_getup_mimic_reset.h"

#include "g1_async_log.h"
#include "g1_control_options.h"
#include "g1_control_runtime.h"
#include "g1_control_signals.h"
#include "g1_control_targets.h"
#include "g1_dds_state_cache.h"
#include "g1_hardware.h"
#include "g1_keyboard.h"
#include "g1_loop_scheduler.h"
#include "g1_monitor_publisher.h"
#include "g1_motion_service.h"
#include "g1_motor_channel.h"
#include "g1_policy_worker.h"
#include "g1_unitree_remote.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using namespace g1_getup_mimic;
using LowState = unitree_hg::msg::dds_::LowState_;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using Clock = std::chrono::steady_clock;
using Receiver = g1_control::DdsStateCache<LowState>;
using Snapshot = g1_control::StateCache<LowState>::Snapshot;

Snapshot WaitForState(const Receiver& receiver, const HardwareSettings& settings) {
    const auto end = Clock::now() + g1_control::kStartupTimeout;
    while (!g1_control::StopRequested() && Clock::now() < end) {
        const auto snapshot = receiver.Read();
        if (snapshot.fault) throw std::runtime_error(snapshot.error);
        if (snapshot.valid && snapshot.AgeMs(Clock::now()) <= settings.limits.timeout_ms) return snapshot;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (g1_control::StopRequested()) return {};
    throw std::runtime_error("No fresh G1 LowState on rt/lowstate. " + receiver.Error());
}

State ReadPolicyState(const LowState& source) {
    State state;
    for (int joint = 0; joint < kJointCount; ++joint) {
        state.position[joint] = source.motor_state()[joint].q();
        state.velocity[joint] = source.motor_state()[joint].dq();
    }
    state.pelvis_quaternion = source.imu_state().quaternion();
    state.pelvis_angular_velocity = source.imu_state().gyroscope();
    return state;
}

void StartReceiver(Receiver& receiver, const HardwareSettings& settings, bool read_only,
                   const std::atomic<bool>& armed, std::atomic<bool>& remote_stop) {
    receiver.Start([&, read_only](const LowState& source) {
        if (armed || read_only) {
            g1_hardware::CheckMotorHealth(source, settings.limits.temperature);
            CheckMeasuredPositions(settings, ReadPolicyState(source));
        } else {
            g1_hardware::CheckTelemetryState(source);
        }
        // Catch a short Select press even while the handover call is in progress.
        if (!read_only && (source.wireless_remote()[2] & (1u << 3))) remote_stop = true;
    }, [&, read_only] { return armed || read_only; });
}

struct Commands {
    LowCmd motor;
    LowCmd monitor;
};

Commands BuildCommands(const HardwareSettings& settings, const State& state, std::uint8_t mode_machine,
                       const MotorTargets* targets, float weight, trace::ControlRecord* record = nullptr) {
    Commands commands{};
    commands.monitor.mode_machine() = mode_machine;
    commands.monitor.mode_pr() = 0;  // PR coordinates.
    const auto actuation = ComputeActuation(settings, state, targets, weight);
    if (record) {
        record->estimated_pd_torque = actuation.torque;
        record->effective_kp = actuation.applied.kp;
        record->effective_kd = actuation.applied.kd;
    }
    g1_control::SetPositionTargets(commands.monitor, actuation.applied.position,
                                   actuation.applied.kp, actuation.applied.kd);
    commands.motor = commands.monitor;
    if (settings.actuation == ActuationMode::HostTorque) {
        for (int joint = 0; joint < kJointCount; ++joint) {
            auto& motor = commands.motor.motor_cmd()[joint];
            motor.kp() = motor.kd() = 0.f;
            motor.tau() = actuation.torque[joint];
        }
    }
    g1_hardware::CheckCommand(commands.motor);
    commands.motor.crc() = g1_hardware::Crc(commands.motor);
    return commands;
}

struct PolicyInput {
    State state;
    std::uint64_t sequence;
    std::uint64_t control_sample_id{}, activation_id{}, submitted_ns{};
    std::uint32_t lowstate_tick{};
};
struct TimedTargets {
    MotorTargets targets;
    double milliseconds;
    std::uint64_t inference_id{};
};

class RealController {
    using Worker = g1_inference::PolicyWorker<PolicyInput, TimedTargets>;

public:
    RealController(const HardwareSettings& settings, const Receiver& receiver, Policy& policy,
                   const PolicyConfig& config, const std::string& reset_pose,
                   std::atomic<bool>& armed, const std::atomic<bool>& remote_stop, trace::Recorder* recorder)
        : settings_(settings), receiver_(receiver), armed_(armed), remote_stop_(remote_stop),
          reset_(config, settings, reset_pose), recorder_(recorder), worker_(
              [this, &policy](const PolicyInput& input, const auto&) {
                  const auto producer_begin = recorder_ ? trace::NowNs() : 0;
                  auto* record = recorder_ ? recorder_->BeginInference() : nullptr;
                  const auto inference_id = ++inference_sequence_;
                  if (record) {
                      record->id = inference_id;
                      record->activation_id = input.activation_id;
                      record->control_sample_id = input.control_sample_id;
                      record->state_received_ns = input.sequence;
                      record->submitted_ns = input.submitted_ns;
                      record->producer_begin_ns = producer_begin;
                      record->lowstate_tick = input.lowstate_tick;
                      record->q = input.state.position;
                      record->dq = input.state.velocity;
                      record->pelvis_quaternion = input.state.pelvis_quaternion;
                      record->pelvis_gyro = input.state.pelvis_angular_velocity;
                  }
                  const auto start = Clock::now();
                  MotorTargets targets;
                  try {
                      targets = policy.Step(input.state, record ? &record->policy : nullptr);
                  } catch (...) {
                      if (record) {
                          record->flags = 2;
                          record->inference_start_ns = trace::Ns(start);
                          record->inference_end_ns = trace::NowNs();
                          record->enqueue_ready_ns = trace::NowNs();
                          recorder_->CommitInference();
                      }
                      throw;
                  }
                  const auto finished = Clock::now();
                  const auto elapsed = finished - start;
                  if (record) {
                      record->flags = 1;
                      record->inference_start_ns = trace::Ns(start);
                      record->inference_end_ns = trace::Ns(finished);
                      record->target = targets.position;
                      record->kp = targets.kp;
                      record->kd = targets.kd;
                      record->enqueue_ready_ns = trace::NowNs();
                      recorder_->CommitInference();
                  }
                  const double milliseconds = std::chrono::duration<double, std::milli>(elapsed).count();
                  if (elapsed > kPolicyPeriod) {
                      log_.Line() << "Policy inference took " << milliseconds << " ms, exceeding the "
                                  << kPolicyPeriod.count() << " ms period.\n";
                  }
                  return TimedTargets{targets, milliseconds, inference_id};
              }, [&policy] { policy.Reset(); }, true, kPolicyPeriod) {}

    int Run() {
        telemetry_.Start();
        telemetry_.PublishEnable(false);
        std::cout << "GetUpMimic DRY RUN. F/A: pose reset, "
                     "P/X: policy, 9/Start: takeover, Space/Select: exit. Reset pose: "
                  << reset_.Selected() << '\n' << std::flush;
        g1_control::LoopScheduler<> cycle;
        bool failed = false;
        try {
            while (!StopRequested()) {
                trace::ControlScope record_scope(recorder_, ++control_sequence_);
                auto* record = record_scope.Get();
                auto snapshot = ReadFresh();
                remote_.Update(snapshot.value.wireless_remote());
                const char remote_action = remote_.state.Action();
                const char action = g1_control::SelectAction(keyboard_.Read(false),
                    remote_action == 'r' ? 'f' : remote_action);  // A resets only this controller.
                if (action == ' ') break;
                if (action == '9' && !armed_) TakeOver(snapshot);
                if (StopRequested()) break;
                const auto state = ReadPolicyState(snapshot.value);
                if (record) {
                    record->flags = trace::StateValid;
                    record->state_received_ns = trace::Ns(snapshot.received_at);
                    record->lowstate_tick = snapshot.value.tick();
                    record->q = state.position;
                    record->dq = state.velocity;
                    record->pelvis_quaternion = state.pelvis_quaternion;
                    record->pelvis_gyro = state.pelvis_angular_velocity;
                    record->pelvis_accelerometer = snapshot.value.imu_state().accelerometer();
                    for (int j=0;j<kJointCount;++j) record->tau_est[j]=snapshot.value.motor_state()[j].tau_est();
                }
                if (armed_) CheckActiveState(snapshot.value, state);
                if (action == 'f' || action == 'F') BeginReset(state);
                if ((action == 'p' || action == 'P') && !running_) Begin();
                if (record) record->activation_id = activation_id_;
                Update(snapshot, state, record);
                record_scope.Finish();
                cycle.Sleep();
            }
        } catch (const std::exception& error) {
            log_.Line() << "GetUpMimic stopped: " << error.what() << '\n';
            failed = true;
        }
        HaltPolicy();
        try {
            if (armed_) ReleaseOutput(failed);
            telemetry_.PublishEnable(false);
        } catch (const std::exception& error) {
            log_.Line() << "GetUpMimic stop output failed: " << error.what() << '\n';
            failed = true;
        }
        return failed ? 1 : 0;
    }

private:
    bool StopRequested() const {
        return g1_control::StopRequested() || remote_stop_;
    }

    Snapshot ReadFresh() const {
        const auto snapshot = receiver_.Read();
        if (!snapshot.valid || snapshot.AgeMs(Clock::now()) > settings_.limits.timeout_ms) {
            throw std::runtime_error("LowState stale or invalid. " + snapshot.error);
        }
        return snapshot;
    }

    void CheckActiveState(const LowState& source, const State& state) const {
        if (monitor_->Conflict()) throw std::runtime_error("Competing LowCmd publisher");
        g1_hardware::CheckMotorHealth(source, settings_.limits.temperature);
        CheckMeasuredPositions(settings_, state);
    }

    void TakeOver(Snapshot& snapshot) {
        if (!g1_control::InputReady(&remote_.state) ||
            !ReadyToGetUp(settings_, ReadPolicyState(snapshot.value))) {
            log_.Line() << "Takeover requires centered sticks and a resting robot.\n";
            return;
        }
        g1_hardware::CheckMotorHealth(snapshot.value, settings_.limits.temperature);
        CheckMeasuredPositions(settings_, ReadPolicyState(snapshot.value));
        HaltPolicy();
        if (StopRequested()) return;
        monitor_ = std::make_unique<g1_control::CommandMonitor>();
        publisher_ = std::make_unique<g1_control::MotorPublisher>("rt/lowcmd");
        publisher_->InitChannel();
        g1_control::ReleaseMotionService();
        monitor_->Arm(*publisher_);
        mode_machine_ = snapshot.value.mode_machine();
        armed_ = true;
        // Before enabling damping, honor any keyboard stop entered during handover
        // and drop queued motion keys.
        for (char key; (key = keyboard_.Read(false));) {
            if (key == ' ') {
                g1_control::RequestStop(0);
                break;
            }
        }
        if (StopRequested()) return;
        snapshot = ReadFresh();
        // Discard an X/P queued on the remote before the handover completed.
        remote_ = g1_unitree_remote::Input{};
        remote_.Update(snapshot.value.wireless_remote());
        if (remote_.state.Action() == ' ') g1_control::RequestStop(0);
        log_.Line() << "Control enabled. Damping only. P/X starts get-up from the measured pose.\n";
    }

    void Begin() {
        // P starts from live state, including during reset and with tracking error.
        if (reset_.Active()) {
            entry_hold_ = reset_.Update(Seconds());
            reset_.Cancel();
        } else entry_hold_.reset();
        ++activation_id_;
        worker_.Begin(false);
        running_ = true;
        started_ = Clock::now();
        ramp_started_.reset();
        next_report_ = started_ + std::chrono::seconds(1);
        latest_reported_ = {};
        peak_inference_ms_ = 0;
        log_.Line() << (armed_ ? "Get-up ON.\n" : "Dry-run policy ON. Targets are preview only.\n");
    }

    void Update(const Snapshot& snapshot, const State& state, trace::ControlRecord* record) {
        const auto now = Clock::now();
        if (reset_.Active()) {
            const bool was_complete = reset_.Complete();
            const auto target = reset_.Update(Seconds());
            if (!was_complete && reset_.Complete()) {
                log_.Line() << "Reset target interpolation complete. Holding; P/X starts policy from live state, F/A repeats reset.\n";
            }
            if (record) record->phase = reset_.Complete() ? trace::Holding : trace::Resetting;
            Publish(state, snapshot, &target, 1.f, record);
            return;
        }
        std::optional<Worker::Result> latest;
        if (running_) {
            worker_.Submit({state, trace::Ns(snapshot.received_at), control_sequence_, activation_id_,
                            recorder_ ? trace::NowNs() : 0, snapshot.value.tick()}, {});
            latest = worker_.Latest();
            if (latest) TrackProgress(*latest, state, now);
        }
        if (record) {
            record->phase = running_ ? (progress_.IsStanding() ? trace::Standing : trace::Policy)
                                     : (armed_ ? trace::Standby : trace::Off);
            if (latest && running_) {
                record->inference_id = latest->target.inference_id;
                record->result_input_ns = trace::Ns(latest->input_at);
                record->result_completed_ns = trace::Ns(latest->completed_at);
            }
        }
        telemetry_.PublishEnable(armed_);
        if (!armed_ && !latest && !record) return;
        float weight = 0.f;
        if (latest && running_) {
            const auto elapsed = std::chrono::duration<double>(now - *ramp_started_).count();
            weight = RampWeight(settings_, elapsed);
        }
        std::optional<MotorTargets> blended;
        const MotorTargets* target = latest && running_ ? &latest->target.targets : nullptr;
        if (entry_hold_ && running_) {
            blended = target ? BlendPoseEntry(*entry_hold_, *target, weight) : *entry_hold_;
            target = &*blended;
            weight = 1.f;
        }
        Publish(state, snapshot, target, weight, record);
    }

    void Publish(const State& state, const Snapshot& snapshot, const MotorTargets* target, float weight,
                 trace::ControlRecord* record) {
        telemetry_.PublishEnable(armed_);
        const auto commands = BuildCommands(settings_, state, snapshot.value.mode_machine(), target, weight, record);
        if (record) {
            record->command_ready_ns = trace::NowNs();
            record->requested_target = target ? target->position : state.position;
            if (target) record->flags |= trace::RequestedTarget;
            RecordCommand(*record, commands.motor);
        }
        if (armed_) {
            if (record) record->flags |= trace::Armed;
            if (StopRequested()) return;
            if (monitor_->Conflict()) throw std::runtime_error("Competing LowCmd publisher");
            if (record) record->write_start_ns = trace::NowNs();
            const bool written = publisher_->Write(commands.motor);
            if (record) {
                record->write_end_ns = trace::NowNs();
                if (written) record->flags |= trace::Published;
            }
            if (!written) throw std::runtime_error("LowCmd publication failed");
            telemetry_.PublishTarget(commands.monitor);
        } else if (running_ || reset_.Active()) {
            telemetry_.PublishPreview(commands.monitor);
            if (record) record->flags |= trace::Preview;
        }
    }

    static void RecordCommand(trace::ControlRecord& record, const LowCmd& command) {
        record.flags |= trace::CommandPrepared;
        for (int j=0;j<kJointCount;++j) {
            const auto& motor=command.motor_cmd()[j];
            record.command_q[j]=motor.q();record.command_dq[j]=motor.dq();
            record.command_kp[j]=motor.kp();record.command_kd[j]=motor.kd();record.command_tau[j]=motor.tau();
        }
    }

    void TrackProgress(const Worker::Result& latest, const State& state, Clock::time_point now) {
        if (!ramp_started_) ramp_started_ = now;
        if (latest.completed_at != latest_reported_) {
            latest_reported_ = latest.completed_at;
            peak_inference_ms_ = std::max(peak_inference_ms_, latest.target.milliseconds);
        }
        if (now >= next_report_) {
            log_.Line() << "Policy inference peak " << peak_inference_ms_ << " ms ("
                        << kPolicyPeriod.count() << " ms period).\n";
            peak_inference_ms_ = 0;
            next_report_ = now + std::chrono::seconds(1);
        }
        if (!armed_ || progress_.IsStanding()) return;
        const auto elapsed = std::chrono::duration<double>(now - started_).count();
        const auto status = progress_.Update(settings_, state, elapsed);
        if (status == RecoveryProgress::Status::Standing) {
            log_.Line() << "Standing. GetUpMimic remains active.\n";
        } else if (status == RecoveryProgress::Status::TimedOut) {
            HaltPolicy();
            log_.Line() << "Get-up timed out. Damping only. P/X retries once the robot has settled.\n";
        }
    }

    void HaltPolicy() {
        worker_.Pause();
        running_ = false;
        progress_.Reset();
        reset_.Cancel();
        entry_hold_.reset();
    }

    static double Seconds() {
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

    void BeginReset(const State& state) {
        if (armed_ && (!g1_control::InputReady(&remote_.state) || !ReadyToGetUp(settings_, state))) {
            log_.Line() << "Pose reset requires centered sticks and a resting robot.\n";
            return;
        }
        HaltPolicy();
        reset_.Begin(state, Seconds());
        log_.Line() << (armed_ ? "REAL" : "PREVIEW") << " pose reset: " << reset_.Selected()
                    << ", duration " << reset_.Duration() << " s; then hold.\n";
    }

    void ReleaseOutput(bool failed) {
        LowCmd stop{};
        stop.mode_pr() = 0;
        stop.mode_machine() = mode_machine_;
        // Manual stop requests zero torque. Faults use drive-side damping without stale PD state.
        g1_control::SetDampingTargets(stop, true, failed ? settings_.limits.stop_kd : 0.f);
        stop.crc() = g1_hardware::Crc(stop);
        const auto end = Clock::now() + std::chrono::duration<double>(settings_.limits.damping_seconds);
        g1_control::LoopScheduler<> cycle;
        bool written = false;
        while (!monitor_->Conflict() && Clock::now() < end) {
            trace::ControlScope scope(recorder_,++control_sequence_);
            auto* record=scope.Get();
            if(record) {
                record->activation_id=activation_id_;record->phase=trace::Stop;
                record->flags=trace::Armed;
                if(failed)record->flags|=trace::FailedStop;
                RecordCommand(*record,stop);record->command_ready_ns=trace::NowNs();
                record->write_start_ns=trace::NowNs();
            }
            const bool sent=publisher_->Write(stop);
            written |= sent;
            if(record) {record->write_end_ns=trace::NowNs();if(sent)record->flags|=trace::Published;}
            scope.Finish();
            cycle.Sleep();
        }
        publisher_.reset();
        armed_ = false;
        if (!written && !monitor_->Conflict()) throw std::runtime_error("No stop command could be published");
    }

    const HardwareSettings& settings_;
    const Receiver& receiver_;
    std::atomic<bool>& armed_;
    const std::atomic<bool>& remote_stop_;
    g1_control::AsyncLog log_;
    g1_control::Keyboard keyboard_;
    g1_unitree_remote::Input remote_;
    g1_control::MonitorPublisher telemetry_;
    std::unique_ptr<g1_control::CommandMonitor> monitor_;
    PoseReset reset_;
    std::optional<MotorTargets> entry_hold_;
    std::unique_ptr<g1_control::MotorPublisher> publisher_;
    trace::Recorder* recorder_;
    std::uint64_t control_sequence_=0, activation_id_=0, inference_sequence_=0;
    Worker worker_;
    Clock::time_point started_{}, next_report_{}, latest_reported_{};
    std::optional<Clock::time_point> ramp_started_;
    RecoveryProgress progress_;
    double peak_inference_ms_ = 0;
    std::uint8_t mode_machine_ = 0;
    bool running_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: g1_getup_mimic_real {read-only|real} ROBOT_INTERFACE [--config JSON] [--expert NAME]"
                         " [--actuation drive_pd|host_torque] [--reset-pose recovered|recorded|3|dwaq-reset] [--record-dir DIRECTORY]\n";
            return 0;
        }
        const std::string mode = argv[1];
        if (argc < 3 || (mode != "read-only" && mode != "real")) {
            throw std::invalid_argument("Use read-only or real and specify the robot interface");
        }
        g1_control::RuntimeOptions options{true, 0, argv[2], G1_GETUP_MIMIC_DEFAULT_CONFIG};
        g1_control::ValidateRuntimeOptions(options);
        std::string expert;
        std::string reset_pose = "recovered";
        std::string record_dir;
        std::optional<ActuationMode> actuation;
        for (int i = 3; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--config") {
                options.config = g1_control::OptionValue(argc, argv, i, key);
            } else if (key == "--expert") {
                expert = g1_control::OptionValue(argc, argv, i, key);
                if (expert.empty()) throw std::invalid_argument("Expected a model_test expert name");
            } else if (key == "--actuation") {
                actuation = ParseActuationMode(g1_control::OptionValue(argc, argv, i, key));
            } else if (key == "--reset-pose") {
                reset_pose = g1_control::OptionValue(argc, argv, i, key);
            } else if (key == "--record-dir" && mode == "real") {
                record_dir = g1_control::OptionValue(argc, argv, i, key);
                if (record_dir.empty()) throw std::invalid_argument("Expected a controller trace directory");
            } else {
                throw std::invalid_argument("Unknown option: " + key);
            }
        }
        const PolicyConfig config(options.config, expert);
        const HardwareSettings settings(config.root, actuation);
        if (!expert.empty()) std::cout << "model_test expert " << expert << std::endl;
        std::cout << "GetUpMimic actuation " << ActuationName(settings.actuation) << std::endl;
        std::unique_ptr<trace::Recorder> recorder;
        if (!record_dir.empty()) {
            trace::Metadata metadata;
            metadata.config_path = std::filesystem::absolute(options.config).string();
            metadata.reset_path = g1_config::ResolvePath(config.path,
                config.root["reset_path"] ? config.root["reset_path"].as<std::string>() : "g1_getup_mimic_reset.json");
            metadata.model_path = g1_config::ResolvePath(config.path,config.model["policy_path"].as<std::string>());
            metadata.reset_pose = reset_pose;metadata.expert = expert;metadata.actuation = ActuationName(settings.actuation);
            metadata.policy_names = config.root["joint_names"].as<std::vector<std::string>>();
            metadata.motor_names.assign(g1_joints::kDdsNames.begin(),g1_joints::kDdsNames.end());
            recorder = std::make_unique<trace::Recorder>(record_dir,metadata);
            std::cout << "Controller trace: " << recorder->Directory() << " (asynchronous, best effort)\n";
        }
        g1_control::InstallSignalHandlers();
        unitree::robot::ChannelFactory::Instance()->Init(options.domain, options.interface);
        std::atomic<bool> armed{false}, remote_stop{false};
        const bool read_only = mode == "read-only";
        Receiver receiver("rt/lowstate");
        StartReceiver(receiver, settings, read_only, armed, remote_stop);
        const auto snapshot = WaitForState(receiver, settings);
        if (g1_control::StopRequested()) return 0;
        const auto state = ReadPolicyState(snapshot.value);
        if (read_only) {
            g1_hardware::CheckMotorHealth(snapshot.value, settings.limits.temperature);
            CheckMeasuredPositions(settings, state);
            const auto gravity = ProjectedGravity(state.pelvis_quaternion);
            std::cout << "GetUpMimic read-only OK. Pelvis gravity [" << gravity[0] << ", " << gravity[1]
                      << ", " << gravity[2] << "]. Resting: "
                      << (ReadyToGetUp(settings, state) ? "yes" : "no") << '\n';
            return 0;
        }
        Policy policy(config);
        policy.Step(state);  // Warm the model without allowing motor output.
        policy.Reset();
        const int result = RealController(settings, receiver, policy, config, reset_pose, armed, remote_stop, recorder.get()).Run();
        // The temporary controller and its inference worker are now destroyed.
        if (recorder) {
            recorder->Close();
            std::cout << "Controller trace closed: " << recorder->Directory() << "; dropped inference="
                      << recorder->InferenceDropped() << ", control=" << recorder->ControlDropped()
                      << ", write errno=" << recorder->Error() << '\n';
        }
        return result;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
