#!/usr/bin/env python3
""" Fake HGGD grasp pose publisher for dry-run test (no camera / no model needed)

    Publish a known grasp target onto /hggd_pose_raw, expressed in the camera
    optical frame, so hggd_endpose_dual_arm.py can transform it back to arm base.
    Print the expected pos_cmd values for manual comparison with script output.

    Usage:
        rosrun piper fake_hggd_pose.py _arm_ns:=arm1                    # default target
        rosrun piper fake_hggd_pose.py _arm_ns:=arm1 _x:=0.90           # out of workspace case
    Params (target is given in ARM BASE frame):
        ~arm_ns    arm namespace, default arm1
        ~cam_frame default {arm_ns}/camera_color_optical_frame, same as the script
        ~x ~y ~z   target position in arm base, default 0.40 0.00 0.25
        ~roll ~pitch ~yaw   target orientation in arm base, default 0 1.483 0 (natural piper pose)
        ~width     gripper width, default 0.03
"""

import json
import numpy as np
import rospy
import tf2_ros
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as SR


def main():
    # init first so that CLI _x:=0.90 style params reach get_param (uploaded in init_node)
    rospy.init_node("fake_hggd_pose", anonymous=True)
    arm_ns = rospy.get_param("~arm_ns", "arm1")
    cam_frame = rospy.get_param(
        "~cam_frame", "%s/camera_color_optical_frame" % arm_ns)
    arm_base_frame = "%s/arm_base" % arm_ns
    target_xyz = [rospy.get_param("~x", 0.40),
                  rospy.get_param("~y", 0.00),
                  rospy.get_param("~z", 0.25)]
    target_rpy = [rospy.get_param("~roll", 0.0),
                  rospy.get_param("~pitch", 1.483),  # ~85deg, natural piper pose
                  rospy.get_param("~yaw", 0.0)]
    width = rospy.get_param("~width", 0.03)

    pub = rospy.Publisher("/hggd_pose_raw", String, queue_size=10, latch=True)

    buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer)
    rospy.loginfo("waiting TF %s <- %s ...", cam_frame, arm_base_frame)
    T = None
    while not rospy.is_shutdown() and T is None:
        try:
            T = buffer.lookup_transform(cam_frame, arm_base_frame, rospy.Time())
        except tf2_ros.TransformException:
            rospy.sleep(0.2)
    if T is None:
        return

    # T maps points from arm base frame to camera frame
    t = np.array([T.transform.translation.x, T.transform.translation.y,
                  T.transform.translation.z])
    q = [T.transform.rotation.x, T.transform.rotation.y,
         T.transform.rotation.z, T.transform.rotation.w]
    R = SR.from_quat(q).as_matrix()

    R_des = SR.from_euler('xyz', target_rpy).as_matrix()
    pos_cam = R @ np.array(target_xyz) + t
    R_cam = R @ R_des

    grasps = [{
        "score": 0.9,
        "width": width,
        "height": 0.02,
        "depth": 0.02,
        "translation": pos_cam.tolist(),
        "rotation": R_cam.tolist(),
    }]
    pub.publish(json.dumps(grasps))

    # print expected values for comparison (approach axis = x column of R_des)
    approach = R_des[:, 0]
    upper_cmd = (np.array(target_xyz) - 0.10 * approach).tolist() + target_rpy
    rospy.loginfo("expected GRIP  pos_cmd: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                  target_xyz[0], target_xyz[1], target_xyz[2],
                  target_rpy[0], target_rpy[1], target_rpy[2])
    rospy.loginfo("expected UPPER pos_cmd (backdist 0.10): [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                  *upper_cmd)
    rospy.loginfo("expected gripper: ready=%.4f close=%.4f",
                  min(width + 0.02, 0.07), min(width, 0.07))
    rospy.sleep(1.0)


if __name__ == '__main__':
    main()
