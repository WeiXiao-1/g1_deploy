"""Forward a pelvis-frame ROS 2 Odometry estimate to the StableMimic DDS input."""

import argparse
import math
import socket

DEFAULT_TOPIC = "rt/stablemimic/base_state"
MAX_SOURCE_AGE_NS = 60_000_000


def pelvis_fields(odometry):
    """Return world position, body-to-world wxyz, and link-origin world velocity."""
    pose, twist = odometry.pose.pose, odometry.twist.twist
    p, quat, linear = pose.position, pose.orientation, twist.linear
    position = [p.x, p.y, p.z]
    attitude = [quat.w, quat.x, quat.y, quat.z]
    velocity = [linear.x, linear.y, linear.z]
    norm = sum(value * value for value in attitude)
    if not 0.5 <= norm <= 1.5 or not all(math.isfinite(v) for v in position + velocity):
        raise ValueError("Invalid pelvis odometry")
    w, x, y, z = [value / math.sqrt(norm) for value in attitude]
    vx, vy, vz = velocity
    world_velocity = [
        (1 - 2*(y*y + z*z))*vx + 2*(x*y - z*w)*vy + 2*(x*z + y*w)*vz,
        2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz,
        2*(x*z - y*w)*vx + 2*(y*z + x*w)*vy + (1 - 2*(x*x + y*y))*vz,
    ]
    return position, [w, x, y, z], world_velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface", help="Robot network interface")
    parser.add_argument("--odom", required=True, help="ROS 2 nav_msgs/msg/Odometry topic")
    parser.add_argument("--world-frame", default="odom", help="Expected gravity-aligned floor-referenced frame")
    parser.add_argument("--pelvis-frame", default="pelvis", help="Expected pelvis link frame")
    parser.add_argument("--dds-topic", default=DEFAULT_TOPIC)
    args = parser.parse_args()
    if args.dds_topic in {"rt/sportmodestate", "rt/odommodestate"}:
        parser.error("Choose a dedicated StableMimic DDS topic")
    socket.if_nametoindex(args.interface)

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    ChannelFactoryInitialize(0, args.interface)
    publisher = ChannelPublisher(args.dds_topic, SportModeState_)
    publisher.Init()
    rclpy.init()
    node = rclpy.create_node("stablemimic_odometry_bridge")
    last_stamp = 0

    def relay(odometry):
        nonlocal last_stamp
        source_time = odometry.header.stamp
        stamp = source_time.sec * 10**9 + source_time.nanosec
        age = node.get_clock().now().nanoseconds - stamp
        try:
            if (odometry.header.frame_id, odometry.child_frame_id) != (args.world_frame, args.pelvis_frame):
                raise ValueError("Unexpected world/pelvis frames")
            if stamp <= last_stamp or not -20_000_000 <= age <= MAX_SOURCE_AGE_NS:
                raise ValueError("Stale, repeated or future-dated odometry")
            position, quaternion, velocity = pelvis_fields(odometry)
        except ValueError as error:
            node.get_logger().warning(str(error), throttle_duration_sec=1.0)
            return
        packet = unitree_go_msg_dds__SportModeState_()
        packet.stamp.sec, packet.stamp.nanosec = source_time.sec, source_time.nanosec
        packet.position, packet.velocity = position, velocity
        packet.imu_state.quaternion = quaternion
        if publisher.Write(packet):
            last_stamp = stamp

    subscription = node.create_subscription(Odometry, args.odom, relay, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
