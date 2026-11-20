#pragma once

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <unistd.h>

namespace g1_getup_mimic::trace {
inline constexpr std::size_t kJoints = 29, kMaxObservation = 744;
using Clock = std::chrono::steady_clock;
inline std::uint64_t Ns(Clock::time_point value) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count();
}
inline std::uint64_t NowNs() { return Ns(Clock::now()); }

struct PolicySnapshot {
    std::uint32_t observation_size{};
    std::array<float, kMaxObservation> observation{};  // Exact ONNX input, policy/term order.
    std::array<float, kJoints> onnx_action{}, applied_action{};  // Policy order, before/after external clip.
};
struct InferenceRecord {
    std::uint64_t id{}, activation_id{}, control_sample_id{}, state_received_ns{}, submitted_ns{};
    std::uint64_t producer_begin_ns{}, inference_start_ns{}, inference_end_ns{}, enqueue_ready_ns{};
    std::uint32_t lowstate_tick{}, flags{};  // Inference flags: 1 success, 2 exception.
    std::array<float,kJoints> q{}, dq{};
    std::array<float,4> pelvis_quaternion{};
    std::array<float,3> pelvis_gyro{};
    PolicySnapshot policy{};
    std::array<float,kJoints> target{}, kp{}, kd{};  // Raw policy targets, DDS order.
};
enum Phase : std::uint32_t { Off=0, Standby=1, Resetting=2, Holding=3, Policy=4, Standing=5, Stop=6 };
enum Flags : std::uint32_t { StateValid=1, Armed=2, CommandPrepared=4, Published=8, Preview=16, FailedStop=32, RequestedTarget=64 };
struct ControlRecord {
    std::uint64_t id{}, activation_id{}, inference_id{}, loop_start_ns{}, loop_end_ns{};
    std::uint64_t state_received_ns{}, result_input_ns{}, result_completed_ns{}, command_ready_ns{};
    std::uint64_t write_start_ns{}, write_end_ns{};
    std::uint32_t lowstate_tick{}, phase{}, flags{};
    std::array<float,kJoints> q{}, dq{}, tau_est{};
    std::array<float,4> pelvis_quaternion{};
    std::array<float,3> pelvis_gyro{}, pelvis_accelerometer{};
    std::array<float,kJoints> requested_target{};  // After entry blend, before hardware clip.
    std::array<float,kJoints> command_q{}, command_dq{}, command_kp{}, command_kd{}, command_tau{};
    std::array<float,kJoints> estimated_pd_torque{};  // Host estimate, not measured drive torque.
    std::array<float,kJoints> effective_kp{}, effective_kd{};  // PD before host_torque zeroes drive gains.
};
static_assert(std::is_trivially_copyable_v<InferenceRecord> && std::is_standard_layout_v<InferenceRecord>);
static_assert(std::is_trivially_copyable_v<ControlRecord> && std::is_standard_layout_v<ControlRecord>);
static_assert(sizeof(float)==4 && std::atomic<std::size_t>::is_always_lock_free);

// One producer and one consumer per queue. No allocation, locks or I/O on producer paths.
template<class T, std::size_t Capacity> class Queue {
    std::unique_ptr<T[]> slots_{new T[Capacity]{}};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
public:
    T* Begin() noexcept {
        const auto head=head_.load(std::memory_order_relaxed);
        if (head-tail_.load(std::memory_order_acquire)>=Capacity) return nullptr;
        auto* slot=&slots_[head%Capacity];
        std::memset(static_cast<void*>(slot),0,sizeof(T));
        return slot;
    }
    void Commit() noexcept { head_.fetch_add(1,std::memory_order_release); }
    bool Pop(T& value) noexcept {
        const auto tail=tail_.load(std::memory_order_relaxed);
        if (tail==head_.load(std::memory_order_acquire)) return false;
        value=slots_[tail%Capacity];
        tail_.store(tail+1,std::memory_order_release);
        return true;
    }
};

inline std::string Quote(const std::string& text) {
    std::ostringstream out; out<<'"';
    for (unsigned char c:text) {
        if (c=='"'||c=='\\') out<<'\\'<<c;
        else if (c<32) out<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<c;
    }
    return out.str()+'"';
}
inline std::string Fingerprint(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read trace provenance file: "+path.string());
    std::uint64_t hash=14695981039346656037ULL;
    std::array<char,65536> buffer{};
    while (file.read(buffer.data(),buffer.size()) || file.gcount())
        for (std::streamsize i=0;i<file.gcount();++i) {hash^=static_cast<unsigned char>(buffer[i]);hash*=1099511628211ULL;}
    if (!file.eof()) throw std::runtime_error("Cannot fingerprint: "+path.string());
    std::ostringstream out;out<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return out.str();
}
struct Metadata {
    std::string config_path, reset_path, model_path, reset_pose, expert, actuation;
    std::vector<std::string> policy_names, motor_names;
};

class Recorder {
    Queue<InferenceRecord,256> inference_;
    Queue<ControlRecord,1024> control_;
    std::atomic<bool> closed_{false};
    std::atomic<int> error_{0};
    std::atomic<std::uint64_t> inference_dropped_{0}, control_dropped_{0};
    std::uint64_t inference_written_=0,control_written_=0;
    std::filesystem::path directory_;
    std::FILE* file_=nullptr;
    std::array<char,65536> file_buffer_{};
    std::thread writer_;

    void Manifest(const Metadata& metadata, std::uint64_t wall_ns, std::uint64_t steady_ns) {
        std::ofstream out(directory_/"manifest.json");
        out.exceptions(std::ios::failbit|std::ios::badbit);
        const std::uint32_t endian=1;
        out<<"{\n\"version\":1,\"byte_order\":"<<Quote(*reinterpret_cast<const unsigned char*>(&endian)?"little":"big")
           <<",\"wall_start_ns\":"<<wall_ns<<",\"monotonic_start_ns\":"<<steady_ns
           <<",\"control_period_ms\":2,\"policy_period_ms\":10,\"inference_queue_capacity\":256,\"control_queue_capacity\":1024,\n"
           <<"\"phase_names\":{\"0\":\"off\",\"1\":\"standby\",\"2\":\"resetting\",\"3\":\"holding\",\"4\":\"policy\",\"5\":\"standing\",\"6\":\"stop\"},\n"
           <<"\"control_flag_bits\":{\"1\":\"state_valid\",\"2\":\"armed\",\"4\":\"command_prepared\",\"8\":\"published\",\"16\":\"preview\",\"32\":\"failed_stop\",\"64\":\"requested_target\"},\n"
           <<"\"config_path\":"<<Quote(metadata.config_path)<<",\"model_path\":"<<Quote(metadata.model_path)
           <<",\"model_fingerprint_fnv1a64\":"<<Quote(Fingerprint(metadata.model_path))
           <<",\"config_fingerprint_fnv1a64\":"<<Quote(Fingerprint(metadata.config_path))
           <<",\"reset_pose\":"<<Quote(metadata.reset_pose)<<",\"expert\":"<<Quote(metadata.expert)
           <<",\"actuation\":"<<Quote(metadata.actuation)<<",\n";
        auto names=[&](const char* key,const std::vector<std::string>& values){out<<Quote(key)<<":[";bool first=true;for(const auto& v:values){if(!first)out<<',';first=false;out<<Quote(v);}out<<"],\n";};
        names("policy_joint_names",metadata.policy_names);names("motor_joint_names",metadata.motor_names);
        out<<"\"notes\":\"ONNX inputs and actions use policy order; all q/targets/commands use motor order. State receive and write times are host times, not motor execution times. Zero-copy enqueue is best effort; discarded or cancelled inference results may have no matching published command.\",\n\"records\":{\n";
        auto field=[&](const char* name,const char* type,std::size_t offset,std::size_t count){out<<"{\"name\":"<<Quote(name)<<",\"type\":"<<Quote(type)<<",\"offset\":"<<offset<<",\"count\":"<<count<<"},\n";};
        out<<"\"1\":{\"name\":\"inference\",\"size\":"<<sizeof(InferenceRecord)<<",\"fields\":[\n";
#define IFIELD(name,type,count) field(#name,type,offsetof(InferenceRecord,name),count)
        IFIELD(id,"u8",1);IFIELD(activation_id,"u8",1);IFIELD(control_sample_id,"u8",1);
        IFIELD(state_received_ns,"u8",1);IFIELD(submitted_ns,"u8",1);IFIELD(producer_begin_ns,"u8",1);
        IFIELD(inference_start_ns,"u8",1);IFIELD(inference_end_ns,"u8",1);IFIELD(enqueue_ready_ns,"u8",1);
        IFIELD(lowstate_tick,"u4",1);IFIELD(flags,"u4",1);IFIELD(q,"f4",29);IFIELD(dq,"f4",29);IFIELD(pelvis_quaternion,"f4",4);IFIELD(pelvis_gyro,"f4",3);
        field("observation_size","u4",offsetof(InferenceRecord,policy)+offsetof(PolicySnapshot,observation_size),1);
        field("observation","f4",offsetof(InferenceRecord,policy)+offsetof(PolicySnapshot,observation),kMaxObservation);
        field("onnx_action","f4",offsetof(InferenceRecord,policy)+offsetof(PolicySnapshot,onnx_action),29);
        field("applied_action","f4",offsetof(InferenceRecord,policy)+offsetof(PolicySnapshot,applied_action),29);
        IFIELD(target,"f4",29);IFIELD(kp,"f4",29);IFIELD(kd,"f4",29);
#undef IFIELD
        out<<"{\"name\":\"_reserved\",\"type\":\"u4\",\"offset\":0,\"count\":0}]},\n";
        out<<"\"2\":{\"name\":\"control\",\"size\":"<<sizeof(ControlRecord)<<",\"fields\":[\n";
#define CFIELD(name,type,count) field(#name,type,offsetof(ControlRecord,name),count)
        CFIELD(id,"u8",1);CFIELD(activation_id,"u8",1);CFIELD(inference_id,"u8",1);
        CFIELD(loop_start_ns,"u8",1);CFIELD(loop_end_ns,"u8",1);CFIELD(state_received_ns,"u8",1);
        CFIELD(result_input_ns,"u8",1);CFIELD(result_completed_ns,"u8",1);CFIELD(command_ready_ns,"u8",1);
        CFIELD(write_start_ns,"u8",1);CFIELD(write_end_ns,"u8",1);
        CFIELD(lowstate_tick,"u4",1);CFIELD(phase,"u4",1);CFIELD(flags,"u4",1);
        CFIELD(q,"f4",29);CFIELD(dq,"f4",29);CFIELD(tau_est,"f4",29);
        CFIELD(pelvis_quaternion,"f4",4);CFIELD(pelvis_gyro,"f4",3);CFIELD(pelvis_accelerometer,"f4",3);
        CFIELD(requested_target,"f4",29);CFIELD(command_q,"f4",29);CFIELD(command_dq,"f4",29);
        CFIELD(command_kp,"f4",29);CFIELD(command_kd,"f4",29);CFIELD(command_tau,"f4",29);CFIELD(estimated_pd_torque,"f4",29);
        CFIELD(effective_kp,"f4",29);CFIELD(effective_kd,"f4",29);
#undef CFIELD
        out<<"{\"name\":\"_reserved\",\"type\":\"u4\",\"offset\":0,\"count\":0}]}}}\n";
        out.close();
    }
    template<class T> bool Write(std::uint32_t type,const T& record) noexcept {
        if(error_) return false;
        const std::uint32_t header[]{type,static_cast<std::uint32_t>(sizeof(T))};
        if(std::fwrite(header,sizeof(header),1,file_)!=1 || std::fwrite(&record,sizeof(record),1,file_)!=1) {
            error_=errno?errno:EIO;return false;
        }
        return true;
    }
    void WriteLoop() noexcept {
        InferenceRecord inference{};ControlRecord control{};
        auto flush_at=Clock::now()+std::chrono::milliseconds(500);
        for(;;) {
            const bool have_inference=inference_.Pop(inference),have_control=control_.Pop(control);
            if(have_inference) {if(Write(1,inference)) ++inference_written_;else ++inference_dropped_;}
            if(have_control) {if(Write(2,control)) ++control_written_;else ++control_dropped_;}
            if(Clock::now()>=flush_at) {
                if(!error_ && std::fflush(file_)!=0) error_=errno?errno:EIO;
                flush_at=Clock::now()+std::chrono::milliseconds(500);
            }
            if(!have_inference && !have_control) {
                if(closed_) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        if(std::fclose(file_)!=0 && !error_) error_=errno?errno:EIO;
        file_=nullptr;
    }
public:
    Recorder(const std::string& parent,const Metadata& metadata) {
        const auto wall=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        const auto steady=NowNs();
        const auto base=std::filesystem::absolute(parent);
        std::filesystem::create_directories(base);
        directory_=base/("run_"+std::to_string(wall)+"_"+std::to_string(getpid()));
        if(!std::filesystem::create_directory(directory_)) throw std::runtime_error("Trace run directory already exists");
        std::filesystem::copy_file(metadata.config_path,directory_/"policy_config.json");
        std::filesystem::copy_file(metadata.reset_path,directory_/"reset_config.json");
        Manifest(metadata,wall,steady);
        file_=std::fopen((directory_/"records.bin").c_str(),"wb");
        if(!file_) throw std::runtime_error("Cannot create controller trace file");
        std::setvbuf(file_,file_buffer_.data(),_IOFBF,file_buffer_.size());
        if(std::fwrite("G1TRACE1",8,1,file_)!=1) {std::fclose(file_);file_=nullptr;throw std::runtime_error("Cannot write controller trace header");}
        try {writer_=std::thread([this]{WriteLoop();});} catch(...) {std::fclose(file_);file_=nullptr;throw;}
    }
    Recorder(const Recorder&)=delete;
    ~Recorder() {Close();}
    const std::filesystem::path& Directory() const {return directory_;}
    int Error() const noexcept {return error_.load(std::memory_order_relaxed);}
    std::uint64_t InferenceDropped() const noexcept {return inference_dropped_.load(std::memory_order_relaxed);}
    std::uint64_t ControlDropped() const noexcept {return control_dropped_.load(std::memory_order_relaxed);}
    InferenceRecord* BeginInference() noexcept {
        auto* slot=(!closed_&&!error_)?inference_.Begin():nullptr;
        if(!slot)++inference_dropped_;
        return slot;
    }
    ControlRecord* BeginControl() noexcept {
        auto* slot=(!closed_&&!error_)?control_.Begin():nullptr;
        if(!slot)++control_dropped_;
        return slot;
    }
    void CommitInference() noexcept {inference_.Commit();}
    void CommitControl() noexcept {control_.Commit();}
    // Call only after both producer threads have stopped. File I/O may block here,
    // after motor release, never inside control or inference loops.
    void Close() noexcept {
        if(closed_.exchange(true))return;
        if(writer_.joinable())writer_.join();
        try {
            std::ofstream out(directory_/"stats.json");
            out<<"{\"inference_written\":"<<inference_written_<<",\"control_written\":"<<control_written_
               <<",\"inference_dropped\":"<<InferenceDropped()<<",\"control_dropped\":"<<ControlDropped()
               <<",\"write_error_errno\":"<<Error()<<"}\n";
        }catch(...) {}
    }
};

class ControlScope {
    Recorder* recorder_;
    ControlRecord* record_;
public:
    ControlScope(Recorder* recorder,std::uint64_t id):recorder_(recorder),record_(nullptr) {
        if(recorder_) {
            const auto start=NowNs();record_=recorder_->BeginControl();
            if(record_) {record_->id=id;record_->loop_start_ns=start;}
        }
    }
    ControlRecord* Get() noexcept {return record_;}
    void Finish() noexcept {
        if(record_) {record_->loop_end_ns=NowNs();recorder_->CommitControl();record_=nullptr;}
    }
    ~ControlScope(){Finish();}
};
} // namespace g1_getup_mimic::trace
