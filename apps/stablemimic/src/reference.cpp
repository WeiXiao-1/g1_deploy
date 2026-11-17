#include "g1_stablemimic_reference.h"

#include "g1_joint_mapping.h"

#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace g1_stablemimic {
namespace {

Rotation Multiply(const Rotation& a, const Rotation& b) {
    Rotation result{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                result[3*i+j] += a[3*i+k] * b[3*k+j];
    return result;
}

template<class Values>
void ReadValues(std::istream& row, Values& values) {
    for (auto& v : values)
        if (!(row >> v) || !std::isfinite(v))
            throw std::runtime_error("Missing or invalid reference value");
}

}  // namespace

Rotation FromQuaternion(const Quaternion& q) {
    auto [w, x, y, z] = q;
    const double norm = w*w + x*x + y*y + z*z;
    if (!std::isfinite(norm) || norm < .5 || norm > 1.5)
        throw std::runtime_error("Invalid orientation quaternion");
    const double s = 2.0 / norm;
    return {1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w),
            s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w),
            s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)};
}

Vector3 Rotate(const Rotation& r, const Vector3& v) {
    return {r[0]*v[0]+r[1]*v[1]+r[2]*v[2],
            r[3]*v[0]+r[4]*v[1]+r[5]*v[2],
            r[6]*v[0]+r[7]*v[1]+r[8]*v[2]};
}

Vector3 Cross(const Vector3& a, const Vector3& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}

void SetTorsoPose(State& state, const Vector3& pelvis_position) {
    const double yaw = state.position[12], roll = state.position[13], pitch = state.position[14];
    const double cy = std::cos(yaw), sy = std::sin(yaw), cr = std::cos(roll), sr = std::sin(roll);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const auto yaw_world = Multiply(state.pelvis_rotation, {cy,-sy,0, sy,cy,0, 0,0,1});
    const auto roll_world = Multiply(yaw_world, {1,0,0, 0,cr,-sr, 0,sr,cr});
    state.torso_rotation = Multiply(roll_world, {cp,0,sp, 0,1,0, -sp,0,cp});
    // The observation anchor is the torso_link origin of the StableMimic training MJCF.
    const auto offset = Rotate(yaw_world, {-0.0039635, 0, 0.044});
    for (int axis = 0; axis < 3; ++axis)
        state.torso_position_world[axis] = pelvis_position[axis] + offset[axis];
}

std::vector<ReferenceFrame> LoadReference(const std::string& path) {
    std::ifstream source(path);
    source.imbue(std::locale::classic());
    std::string line, format;
    int fps = 0;
    if (!std::getline(source, line)) throw std::runtime_error("Cannot read reference " + path);
    std::istringstream header(line);
    if (!(header >> format >> fps) || format != "g1-stablemimic-reference-v1" || fps != 50)
        throw std::runtime_error("Expected a 50 Hz g1-stablemimic-reference-v1 file: " + path);
    std::vector<std::string> names;
    for (std::string name; header >> name;) names.push_back(name);
    const auto order = g1_joints::FromNames(names);

    std::vector<ReferenceFrame> frames;
    while (std::getline(source, line)) {
        std::istringstream row(line);
        row.imbue(std::locale::classic());
        try {
            ReferenceFrame frame;
            Joints q{}, dq{};
            Quaternion torso{};
            ReadValues(row, frame.pelvis_position);
            ReadValues(row, frame.command.torso_position_world);
            ReadValues(row, torso);
            ReadValues(row, q);
            ReadValues(row, dq);
            std::string extra;
            if (row >> extra) throw std::runtime_error("Unexpected field");
            frame.command.torso_rotation = FromQuaternion(torso);
            frame.command.position = g1_joints::ToMotorOrder(q, order);
            frame.command.velocity = g1_joints::ToMotorOrder(dq, order);
            frames.push_back(frame);
        } catch (const std::exception& error) {
            throw std::runtime_error(path + " frame " + std::to_string(frames.size()) + ": " + error.what());
        }
    }
    if (source.bad() || frames.empty()) throw std::runtime_error("Unreadable or empty reference: " + path);
    return frames;
}

Playback::Playback(ReferenceFrame standing, std::vector<ReferenceFrame> motion, std::size_t first_frame)
    : standing_(std::move(standing)), motion_(std::move(motion)), first_frame_(first_frame), cursor_(first_frame) {
    if (first_frame_ >= (motion_.empty() ? 1 : motion_.size()))
        throw std::runtime_error("Reference start frame out of range");
}

void Playback::Reset() {
    cursor_ = first_frame_;
    aligned_ = fallen_ = holding_ = false;
    upright_ticks_ = 0;
}

Playback::Sample Playback::Step(const State& state, const Vector3& pelvis_position) {
    const bool holding = cursor_ == motion_.size();
    const auto& frame = holding ? standing_ : motion_[cursor_];
    const double height = pelvis_position[2];
    const double vertical = state.pelvis_rotation[8];
    if (height < .50 || vertical < .5) fallen_ = true;
    upright_ticks_ = fallen_ && height >= .65 && vertical >= std::cos(35.0 * std::acos(-1.) / 180.)
                   ? upright_ticks_ + 1 : 0;
    const bool realigned = upright_ticks_ >= 15;
    if (!aligned_ || realigned || (holding && !holding_)) {
        for (int axis = 0; axis < 2; ++axis)
            offset_[axis] = pelvis_position[axis] - frame.pelvis_position[axis];
        aligned_ = true;
    }
    if (realigned) {
        fallen_ = false;
        upright_ticks_ = 0;
    }
    holding_ = holding;
    Sample sample{frame.command, cursor_, realigned, holding};
    for (int axis = 0; axis < 2; ++axis) sample.command.torso_position_world[axis] += offset_[axis];
    if (!holding) ++cursor_;
    return sample;
}

}  // namespace g1_stablemimic
