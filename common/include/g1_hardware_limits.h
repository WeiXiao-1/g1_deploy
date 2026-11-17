#pragma once

#include "g1_config_utils.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace g1_hardware {
struct LimitsConfig {
    std::array<float, 29> lower{}, upper{};
    double timeout_ms = 100, max_tilt = 1.0, damping_seconds = 0.5;
    float temperature = 70, stop_kd = 1;
    bool limit_target_position = false;
    explicit LimitsConfig(const YAML::Node& root, bool use_tilt_limit = true) {
        const auto cfg = root["hardware"];
        if (!cfg) throw std::runtime_error("Missing hardware configuration");
        lower = g1_config::ReadArray<float, 29>(cfg["joint_lower"]);
        upper = g1_config::ReadArray<float, 29>(cfg["joint_upper"]);
        limit_target_position = cfg["limit_target_position"] ? cfg["limit_target_position"].as<bool>() : false;
        timeout_ms = cfg["state_timeout_ms"].as<double>();
        if (use_tilt_limit) max_tilt = cfg["max_tilt_rad"].as<double>();
        temperature = cfg["max_temperature_c"].as<float>();
        stop_kd = cfg["stop_kd"].as<float>();
        damping_seconds = cfg["damping_seconds"].as<double>();
        for (double v : {timeout_ms, max_tilt, double(temperature), double(stop_kd), damping_seconds})
            if (!std::isfinite(v) || v <= 0) throw std::runtime_error("Invalid hardware protection setting");
        if (timeout_ms > 500 || max_tilt >= 1.57 || damping_seconds > 5)
            throw std::runtime_error("Hardware protection setting outside supported range");
        for (int i = 0; i < 29; ++i)
            if (lower[i]+0.02f >= upper[i])
                throw std::runtime_error("Invalid per-joint hardware limits");
    }
};
} // namespace g1_hardware
