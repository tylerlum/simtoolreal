import copy
import sys
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState


CURRENT_JOINT_POS_IIWA = None
CURRENT_JOINT_POS_WUJI = None

HOME_JOINT_POS_IIWA = np.array(
    [
        -1.571,
        1.571 - np.deg2rad(10),
        -0.000,
        1.376 + np.deg2rad(10),
        -0.000,
        1.485,
        1.308,
    ]
)
HOME_JOINT_POS_WUJI = np.zeros(20)
HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_WUJI])


def current_joint_pos_iiwa_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_IIWA
    CURRENT_JOINT_POS_IIWA = np.array(msg.position)


def current_joint_pos_wuji_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_WUJI
    CURRENT_JOINT_POS_WUJI = np.array(msg.position)


def interpolate_joint_pos(
    init_joint_pos: np.ndarray, final_joint_pos: np.ndarray, num_steps: int
) -> np.ndarray:
    joint_positions_list = []
    for i in range(num_steps):
        joint_positions_list.append(
            init_joint_pos + (final_joint_pos - init_joint_pos) * (i + 1) / num_steps
        )
    return np.array(joint_positions_list)


def publish_joint_pos_targets(
    joint_pos_targets: np.ndarray,
    pub_iiwa: rospy.Publisher,
    pub_wuji: rospy.Publisher,
) -> None:
    iiwa_msg = JointState()
    iiwa_msg.header.stamp = rospy.Time.now()
    iiwa_msg.name = [f"iiwa_joint_{i}" for i in range(1, 8)]
    iiwa_msg.position = copy.deepcopy(joint_pos_targets[:7].tolist())

    wuji_msg = JointState()
    wuji_msg.header.stamp = rospy.Time.now()
    wuji_msg.name = [
        f"left_finger{finger}_joint{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    ]
    wuji_msg.position = copy.deepcopy(joint_pos_targets[7:].tolist())

    pub_iiwa.publish(iiwa_msg)
    pub_wuji.publish(wuji_msg)


def move_to_pose(
    target_pos: np.ndarray,
    pub_iiwa: rospy.Publisher,
    pub_wuji: rospy.Publisher,
    move_time: float = 10.0,
    control_hz: int = 60,
) -> None:
    current_pos = np.concatenate([CURRENT_JOINT_POS_IIWA.copy(), CURRENT_JOINT_POS_WUJI.copy()])
    interpolated_targets = interpolate_joint_pos(
        init_joint_pos=current_pos,
        final_joint_pos=target_pos,
        num_steps=int(control_hz * move_time),
    )
    for target in interpolated_targets:
        if rospy.is_shutdown():
            sys.exit(0)
        start_time = rospy.Time.now()
        publish_joint_pos_targets(target, pub_iiwa=pub_iiwa, pub_wuji=pub_wuji)
        loop_dt = (rospy.Time.now() - start_time).to_sec()
        sleep_dt = 1 / control_hz - loop_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)


def main():
    rospy.init_node("home_wuji_robot", anonymous=True)

    _sub_iiwa = rospy.Subscriber(
        "/iiwa/joint_states", JointState, current_joint_pos_iiwa_callback, queue_size=1
    )
    _sub_wuji = rospy.Subscriber(
        "/wuji/joint_states", JointState, current_joint_pos_wuji_callback, queue_size=1
    )
    pub_iiwa = rospy.Publisher("/iiwa/joint_cmd", JointState, queue_size=1)
    pub_wuji = rospy.Publisher("/wuji/joint_cmd", JointState, queue_size=1)

    while not rospy.is_shutdown():
        if CURRENT_JOINT_POS_IIWA is None or CURRENT_JOINT_POS_WUJI is None:
            rospy.sleep(0.1)
        else:
            break

    move_to_pose(HOME_JOINT_POS, pub_iiwa=pub_iiwa, pub_wuji=pub_wuji)


if __name__ == "__main__":
    main()
