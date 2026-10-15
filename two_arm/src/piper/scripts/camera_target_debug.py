#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_target_debug.py

用于调观察姿态：
- 显示相机当前 optical +Z 光轴
- 显示“相机当前位置 → 目标点”的理想视线方向
- 打印二者夹角，角度越小，说明相机越对准目标

RViz:
  Fixed Frame: arm_base
  Add -> Marker -> /camera_optical_axis_marker
  Add -> Marker -> /camera_target_ray_marker
  Add -> Marker -> /target_point_marker

典型用法：
  rosrun piper camera_target_debug.py _target_x:=0.35 _target_y:=0.0 _target_z:=0.30

说明：
- target_x/y/z 是 arm_base 坐标系下希望相机看到的目标点。
- RealSense optical frame 通常沿 +Z 方向看出去。
"""

import math
import numpy as np
import rospy
import tf2_ros
from scipy.spatial.transform import Rotation as SR
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class CameraTargetDebug:
    def __init__(self):
        self.base_frame = rospy.get_param("~base_frame", "arm_base")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        self.target = np.array([
            float(rospy.get_param("~target_x", 0.35)),
            float(rospy.get_param("~target_y", 0.0)),
            float(rospy.get_param("~target_z", 0.30)),
        ], dtype=float)
        self.arrow_length = float(rospy.get_param("~arrow_length", 0.35))
        self.log_period = float(rospy.get_param("~log_period", 1.0))

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.optical_pub = rospy.Publisher("/camera_optical_axis_marker", Marker, queue_size=1)
        self.target_ray_pub = rospy.Publisher("/camera_target_ray_marker", Marker, queue_size=1)
        self.target_pub = rospy.Publisher("/target_point_marker", Marker, queue_size=1)

        rospy.Timer(rospy.Duration(0.2), self.publish_markers)
        rospy.Timer(rospy.Duration(self.log_period), self.log_state)

        rospy.loginfo(
            "camera_target_debug started. target in %s = (%.3f, %.3f, %.3f)",
            self.base_frame, self.target[0], self.target[1], self.target[2]
        )

    def get_camera_pose_axis(self):
        tf = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.camera_frame,
            rospy.Time(0),
            rospy.Duration(0.5)
        )
        p = tf.transform.translation
        q = tf.transform.rotation
        cam_pos = np.array([p.x, p.y, p.z], dtype=float)
        rot = SR.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        # RealSense optical frame: +Z is forward
        optical_z = rot[:, 2]
        optical_z = optical_z / max(np.linalg.norm(optical_z), 1e-9)
        return cam_pos, optical_z

    def make_arrow(self, ns, mid, start, direction, length, rgba):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.025
        m.scale.y = 0.050
        m.scale.z = 0.080
        m.color.r, m.color.g, m.color.b, m.color.a = rgba

        end = start + direction * length
        m.points = [
            Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
            Point(x=float(end[0]), y=float(end[1]), z=float(end[2])),
        ]
        return m

    def make_target_sphere(self):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = rospy.Time.now()
        m.ns = "target_point"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.target[0])
        m.pose.position.y = float(self.target[1])
        m.pose.position.z = float(self.target[2])
        m.pose.orientation.w = 1.0
        m.scale.x = 0.06
        m.scale.y = 0.06
        m.scale.z = 0.06
        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 0.1
        m.color.a = 0.9
        return m

    def compute_alignment(self):
        cam_pos, optical_z = self.get_camera_pose_axis()
        vec = self.target - cam_pos
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            desired = np.array([0.0, 0.0, 1.0])
            angle = 0.0
        else:
            desired = vec / dist
            dot = float(np.clip(np.dot(optical_z, desired), -1.0, 1.0))
            angle = math.degrees(math.acos(dot))
        return cam_pos, optical_z, desired, dist, angle

    def publish_markers(self, event=None):
        try:
            cam_pos, optical_z, desired, dist, angle = self.compute_alignment()
        except Exception:
            return

        self.optical_pub.publish(
            self.make_arrow(
                "camera_optical_axis", 0,
                cam_pos, optical_z, self.arrow_length,
                (0.1, 0.7, 1.0, 1.0)
            )
        )
        self.target_ray_pub.publish(
            self.make_arrow(
                "camera_target_ray", 0,
                cam_pos, desired, min(max(dist, 0.15), 0.8),
                (0.1, 1.0, 0.2, 1.0)
            )
        )
        self.target_pub.publish(self.make_target_sphere())

    def log_state(self, event=None):
        try:
            cam_pos, optical_z, desired, dist, angle = self.compute_alignment()
            rospy.loginfo(
                "Target debug: cam=(%.3f, %.3f, %.3f), optical+Z=(%.3f, %.3f, %.3f), "
                "desired=(%.3f, %.3f, %.3f), dist=%.3f, angle_error=%.1f deg",
                cam_pos[0], cam_pos[1], cam_pos[2],
                optical_z[0], optical_z[1], optical_z[2],
                desired[0], desired[1], desired[2],
                dist, angle
            )
        except Exception as e:
            rospy.logwarn_throttle(
                3.0,
                "Waiting for TF %s -> %s: %s",
                self.base_frame, self.camera_frame, e
            )


if __name__ == "__main__":
    rospy.init_node("camera_target_debug")
    CameraTargetDebug()
    rospy.spin()
