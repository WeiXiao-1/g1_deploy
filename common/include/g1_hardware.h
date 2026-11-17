#pragma once

#include "g1_hardware_limits.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

namespace g1_hardware {
// Same CRC algorithm and wire-object layout used by the official G1 SDK example.
template<class Message> uint32_t Crc(const Message& message) {
    static_assert(sizeof(Message) % 4 == 0);
    const auto* bytes = reinterpret_cast<const unsigned char*>(&message);
    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < sizeof(Message)/4-1; ++i) {
        uint32_t word;
        std::memcpy(&word, bytes+i*4, 4);
        for (uint32_t mask = 0x80000000u; mask; mask >>= 1) {
            const bool high = (crc & 0x80000000u) != 0;
            crc <<= 1;
            if (high) crc ^= 0x04C11DB7u;
            if (word & mask) crc ^= 0x04C11DB7u;
        }
    }
    return crc;
}
// Integrity checks for previewing real state, without enforcing actuator readiness.
inline void CheckTelemetryState(const unitree_hg::msg::dds_::LowState_& state) {
    if (state.crc() != Crc(state)) throw std::runtime_error("LowState CRC mismatch");
    if (state.mode_machine() != 5) throw std::runtime_error("Hardware requires unlocked G1 29DOF mode_machine=5");
    for (int i = 0; i < 29; ++i) {
        const auto& motor = state.motor_state()[i];
        if (!std::isfinite(motor.q()) || !std::isfinite(motor.dq()) || !std::isfinite(motor.tau_est()))
            throw std::runtime_error("Non-finite motor state");
    }
    double norm = 0;
    for (float v : state.imu_state().quaternion()) norm += double(v)*v;
    if (!std::isfinite(norm) || norm < 0.5 || norm > 1.5) throw std::runtime_error("Invalid hardware IMU quaternion");
    for (float v : state.imu_state().gyroscope())
        if (!std::isfinite(v)) throw std::runtime_error("Invalid hardware gyroscope");
}
// Validate outgoing fields without changing policy/reset targets.
inline void CheckCommand(const unitree_hg::msg::dds_::LowCmd_& cmd) {
    for (int i = 0; i < 29; ++i) {
        const auto& motor = cmd.motor_cmd()[i];
        if (!std::isfinite(motor.q()) || !std::isfinite(motor.dq()) ||
            !std::isfinite(motor.kp()) || !std::isfinite(motor.kd()) || !std::isfinite(motor.tau()))
            throw std::runtime_error("Non-finite hardware command at DDS joint "+std::to_string(i));
    }
}
inline void CheckMotorHealth(const unitree_hg::msg::dds_::LowState_& state, float temperature) {
    CheckTelemetryState(state);
    for (int i = 0; i < 29; ++i) {
        const auto& motor = state.motor_state()[i];
        if (motor.motorstate() != 0) throw std::runtime_error("Motor fault at DDS joint "+std::to_string(i));
        for (auto t : motor.temperature())
            if (t >= temperature) throw std::runtime_error("Motor overtemperature at DDS joint "+std::to_string(i));
    }
}

struct Limits : LimitsConfig {
    using LimitsConfig::LimitsConfig;
    void CheckMotorHealth(const unitree_hg::msg::dds_::LowState_& state) const {
        g1_hardware::CheckMotorHealth(state, temperature);
    }
    void CheckState(const unitree_hg::msg::dds_::LowState_& state, unsigned remote_stop_mask = 1u << 8) const {
        CheckMotorHealth(state);
        double norm = 0;
        for (float v : state.imu_state().quaternion()) norm += double(v)*v;
        const auto& q = state.imu_state().quaternion();
        const double upright = 1 - 2*(double(q[1])*q[1]+double(q[2])*q[2])/norm;
        if (upright < std::cos(max_tilt)) throw std::runtime_error("Hardware tilt limit");
        // Official gamepad byte layout: 2 header bytes, then little-endian key bits.
        const auto& remote = state.wireless_remote();
        const unsigned keys = unsigned(remote[2]) | (unsigned(remote[3]) << 8);
        if (keys & remote_stop_mask) throw std::runtime_error("Remote requested damping stop");
    }
    // Optional position clamp only; no target slew-rate limiting.
    void PrepareCommand(unitree_hg::msg::dds_::LowCmd_& cmd) const {
        CheckCommand(cmd);
        if (limit_target_position) {
            for (int i = 0; i < 29; ++i)
                cmd.motor_cmd()[i].q() = std::clamp(cmd.motor_cmd()[i].q(), lower[i], upper[i]);
        }
    }
};
} // namespace g1_hardware
