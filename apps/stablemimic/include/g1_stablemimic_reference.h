#pragma once

#include "g1_stablemimic_policy.h"

#include <string>
#include <vector>

namespace g1_stablemimic {

using Quaternion = std::array<double, 4>;  // wxyz.
Rotation FromQuaternion(const Quaternion& q);
Vector3 Rotate(const Rotation& r, const Vector3& v);
Vector3 Cross(const Vector3& a, const Vector3& b);
void SetTorsoPose(State& state, const Vector3& pelvis_position);

struct ReferenceFrame {
    Reference command;
    Vector3 pelvis_position;
};

std::vector<ReferenceFrame> LoadReference(const std::string& path);
ReferenceFrame LoadStandingReference(const std::string& path);

class Playback {
public:
    struct Sample {
        Reference command;
        std::size_t frame;
        bool realigned, holding;
    };

    Playback(ReferenceFrame standing, std::vector<ReferenceFrame> motion, std::size_t first_frame);
    void Reset();
    Sample Step(const State& state, const Vector3& pelvis_position);

private:
    ReferenceFrame standing_;
    std::vector<ReferenceFrame> motion_;
    std::size_t first_frame_, cursor_;
    std::array<double, 2> offset_{};
    int upright_ticks_ = 0;
    bool aligned_ = false, fallen_ = false, holding_ = false;
};

}  // namespace g1_stablemimic
