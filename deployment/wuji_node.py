import argparse
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from termcolor import colored

import wujihandpy

from isaacgymenvs.utils.robot_info import WUJI_SPEC


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class WujiRosNode:
    def __init__(self, serial_number: str | None = None):
        rospy.init_node("wuji_ros_node")

        self.rate_hz = 100
        self.rate = rospy.Rate(self.rate_hz)
        self.latest_cmd: np.ndarray | None = None

        self.joint_cmd_sub = rospy.Subscriber(
            "/wuji/joint_cmd", JointState, self.joint_cmd_callback, queue_size=1
        )
        self.joint_states_pub = rospy.Publisher(
            "/wuji/joint_states", JointState, queue_size=1
        )

        info("Connecting to WUJI hand")
        if serial_number:
            self.hand = wujihandpy.Hand(serial_number=serial_number)
        else:
            self.hand = wujihandpy.Hand()

        self.hand.write_joint_enabled(True)
        self.hand.write_joint_target_position(np.zeros((5, 4), dtype=np.float64))

    def joint_cmd_callback(self, msg: JointState):
        q = np.asarray(msg.position, dtype=np.float64)
        expected = WUJI_SPEC.num_hand_dofs
        if q.shape != (expected,):
            warn(f"Ignoring command with shape {q.shape}, expected ({expected},)")
            return
        self.latest_cmd = q

    def publish_joint_states(self):
        q = np.asarray(self.hand.read_joint_actual_position(), dtype=np.float64).reshape(
            -1
        )
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(WUJI_SPEC.joint_names[7:])
        msg.position = q.tolist()
        msg.velocity = np.zeros_like(q).tolist()
        self.joint_states_pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            if self.latest_cmd is not None:
                self.hand.write_joint_target_position(self.latest_cmd.reshape(5, 4))
            self.publish_joint_states()
            self.rate.sleep()

    def stop(self):
        info("Stopping WUJI hand")
        try:
            self.hand.write_joint_enabled(False)
        except Exception as exc:  # pragma: no cover
            warn(f"Failed to disable WUJI joints cleanly: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-number", default=None)
    args = parser.parse_args()

    node = WujiRosNode(serial_number=args.serial_number)
    try:
        node.run()
    finally:
        time.sleep(0.1)
        node.stop()


if __name__ == "__main__":
    main()
