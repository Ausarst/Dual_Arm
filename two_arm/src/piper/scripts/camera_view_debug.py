#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_view_debug.py

在 RViz 中显示 camera_color_optical_frame 的光轴方向，并定期打印相机位置/光轴。
RealSense optical frame 通常沿 +Z 方向看出去。

RViz 设置：
- Fixed Frame: arm_base
- Add -> Marker -> Topic: /camera_view_marker
"""

import rospy
import tf2_ros
from scipy.spatial.transform import Rotation as SR
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class CameraViewDebug:
    def __init__(self):
        self.base_frame = rospy.get_param("~base_frame", "arm_base")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        self.arrow_length = float(rospy.get_param("~arrow_length", 0.35))

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.marker_pub = rospy.Publisher("/camera_view_marker", Marker, queue_size=1)

        rospy.Timer(rospy.Duration(0.2), self.publish_marker)
        rospy.Timer(rospy.Duration(1.0), self.log_axis)
        rospy.loginfo("camera_view_debug started: %s -> %s", self.base_frame, self.camera_frame)

    def get_pose_axis(self):
        tf = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.camera_frame,
            rospy.Time(0),
            rospy.Duration(0.5)
        )
        p = tf.transform.translation
        q = tf.transform.rotation
        rot = SR.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        optical_z = rot[:, 2]
        return p, optical_z

    def publish_marker(self, event=None):
        try:
            p, axis = self.get_pose_axis()
        except Exception:
            return

        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = rospy.Time.now()
        m.ns = "camera_view"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.025
        m.scale.y = 0.050
        m.scale.z = 0.080
        m.color.r = 0.1
        m.color.g = 0.7
        m.color.b = 1.0
        m.color.a = 1.0

        start = Point(x=p.x, y=p.y, z=p.z)
        end = Point(
            x=p.x + axis[0] * self.arrow_length,
            y=p.y + axis[1] * self.arrow_length,
            z=p.z + axis[2] * self.arrow_length,
        )
        m.points = [start, end]
        self.marker_pub.publish(m)

    def log_axis(self, event=None):
        try:
            p, axis = self.get_pose_axis()
            rospy.loginfo(
                "Camera in %s: xyz=(%.3f, %.3f, %.3f), optical+Z=(%.3f, %.3f, %.3f)",
                self.base_frame, p.x, p.y, p.z, axis[0], axis[1], axis[2]
            )
        except Exception as e:
            rospy.logwarn_throttle(3.0, "Waiting for TF %s -> %s: %s",
                                   self.base_frame, self.camera_frame, e)


if __name__ == "__main__":
    rospy.init_node("camera_view_debug")
    CameraViewDebug()
    rospy.spin()
