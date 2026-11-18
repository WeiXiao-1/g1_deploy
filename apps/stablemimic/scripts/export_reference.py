"""Pack a preprocessed StableMimic 50 Hz NPZ into the onboard reference format."""

import argparse
from pathlib import Path

def name_string(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def export_reference(source, output, standing):
    import numpy as np

    with np.load(source, allow_pickle=False) as motion:
        if float(motion["fps"].item()) != 50:
            raise ValueError("Expected a 50 Hz StableMimic NPZ. Resample in the training project first.")
        joints = [name_string(value) for value in motion["joint_names"]]
        bodies = [name_string(value) for value in motion["body_names"]]
        if len(joints) != 29 or len(set(joints)) != 29 or len(set(bodies)) != len(bodies):
            raise ValueError("Expected 29 unique joints and unique body names")
        pelvis, torso = bodies.index("pelvis"), bodies.index("torso_link")
        q, dq = motion["joint_pos"], motion["joint_vel"]
        body_pos, body_quat = motion["body_pos_w"], motion["body_quat_w"]
        frames = len(q)
        if not frames or q.shape != (frames, 29) or dq.shape != q.shape:
            raise ValueError("Invalid joint reference shape")
        if body_pos.shape != (frames, len(bodies), 3) or body_quat.shape != (frames, len(bodies), 4):
            raise ValueError("Invalid body reference shape")
        quat = body_quat[:, torso]
        order = name_string(motion["quat_order"].item()).lower()
        if order == "xyzw":
            quat = quat[:, [3, 0, 1, 2]]
        elif order != "wxyz":
            raise ValueError("Expected xyzw or wxyz quaternions")
        data = np.column_stack((body_pos[:, pelvis], body_pos[:, torso], quat, q, dq))
        norms = (quat * quat).sum(axis=1)
        if not np.isfinite(data).all() or not ((0.5 <= norms) & (norms <= 1.5)).all():
            raise ValueError("Invalid reference data")
        if standing:
            if not np.allclose(data, data[0], atol=1e-6, rtol=0) or not np.allclose(dq, 0, atol=1e-6):
                raise ValueError("A standing hold requires a constant reference with zero joint velocity")
            data = data[:1]

    with output.open("x", encoding="utf-8") as stream:
        stream.write("g1-stablemimic-reference-v1 50 " + " ".join(joints) + "\n")
        np.savetxt(stream, data, fmt="%.9g")
    return len(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--standing", action="store_true", help="Export one static standing frame")
    args = parser.parse_args()
    frames = export_reference(args.npz, args.output, args.standing)
    print(f"Exported {frames} frames at 50 Hz to {args.output}")
