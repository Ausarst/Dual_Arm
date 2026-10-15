#!/usr/bin/env python3
import rospy
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import json
import numpy as np
from scipy.spatial.transform import Rotation as SR
import tf.transformations as transformations
from tf2_geometry_msgs import do_transform_pose

class GraspSimNode:
    def __init__(self):
        rospy.init_node("grasp_sim_node")

        # 先创建发布者，提前发送关节状态激活 robot_state_publisher
        self.js_pub = rospy.Publisher("/joint_states_hggd", JointState, queue_size=10)
        self.current_joints = [0.0] * 6
        self.width = 0.04
        for _ in range(10):
            self._publish_js()
            rospy.sleep(0.1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.traj_points = []
        self.executing = False
        self.traj_start_time = None
        self.traj_duration = 5.0

        # 等待 TF 树稳定
        rospy.loginfo("Waiting for TF tree to include link6...")
        self.root_frame = "arm_base"   # 直接使用 arm_base，不再检测
        # 简单验证 link6 是否存在
        while not rospy.is_shutdown():
            try:
                self.tf_buffer.lookup_transform("arm_base", "link6", rospy.Time(0), rospy.Duration(0.5))
                break
            except:
                self._publish_js()
                rospy.sleep(0.5)
        rospy.loginfo("link6 TF available. Root frame: arm_base")

        # 现在才订阅消息，避免过早收到消息导致 root_frame 为 None
        rospy.Subscriber("/hggd_pose_raw", String, self.grasp_cb, queue_size=1)
        rospy.Subscriber("/joint_states_single", JointState, self.js_cb, queue_size=1)

        rospy.loginfo("Grasp Simulation Node Ready. Root: %s", self.root_frame)

    def get_link6_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(self.root_frame, "link6", rospy.Time(0), rospy.Duration(0.2))
            return trans.transform
        except:
            return None

    def js_cb(self, msg):
        if self.executing:   # 轨迹执行中，不受外界干扰
            return
        if len(msg.position) >= 6:
            self.current_joints = list(msg.position[:6])
            self.broadcast_visual_end_effector()

    def broadcast_visual_end_effector(self):
        tf = self.get_link6_pose()
        if tf is None:
            return
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.root_frame
        t.child_frame_id = "grasp_base_visual"
        t.transform = tf
        self.tf_broadcaster.sendTransform(t)

    def grasp_cb(self, msg):
        if self.root_frame is None:
            rospy.logwarn("Root frame not ready, ignoring grasp message")
            return

        rospy.loginfo("Received grasp message")
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logerr("Invalid JSON: %s", e)
            return
        if not data:
            return

        best = max(data, key=lambda g: g["score"])
        self.width = best.get("width", 0.04)

        tx, ty, tz = best["translation"]
        r = best["rotation"]
        quat = transformations.quaternion_from_matrix([
            [r[0][0], r[0][1], r[0][2], 0],
            [r[1][0], r[1][1], r[1][2], 0],
            [r[2][0], r[2][1], r[2][2], 0],
            [0, 0, 0, 1]
        ])

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "camera_color_optical_frame"
        pose.pose.position.x = tx
        pose.pose.position.y = ty
        pose.pose.position.z = tz
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        try:
            tf = self.tf_buffer.lookup_transform(
                self.root_frame, pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            pose_root = do_transform_pose(pose, tf)
        except Exception as e:
            rospy.logerr("TF transform failed: %s", e)
            return

        self._broadcast_grasp_target(pose_root)
        rospy.loginfo("Target published, starting demo motion")
        self._demo_motion()

    def _broadcast_grasp_target(self, pose_msg):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.root_frame
        t.child_frame_id = "grasp_target"
        t.transform.translation.x = pose_msg.pose.position.x
        t.transform.translation.y = pose_msg.pose.position.y
        t.transform.translation.z = pose_msg.pose.position.z
        t.transform.rotation = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)

        rot_mat = SR.from_quat([pose_msg.pose.orientation.x,
                                pose_msg.pose.orientation.y,
                                pose_msg.pose.orientation.z,
                                pose_msg.pose.orientation.w]).as_matrix()
        back_vec = -0.1 * rot_mat[:, 0]
        tb = geometry_msgs.msg.TransformStamped()
        tb.header.stamp = rospy.Time.now()
        tb.header.frame_id = "grasp_target"
        tb.child_frame_id = "visual_backward"
        tb.transform.translation.x = back_vec[0]
        tb.transform.translation.y = back_vec[1]
        tb.transform.translation.z = back_vec[2]
        tb.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tb)

    def _demo_motion(self):
        start_j1 = self.current_joints[0]
        target_j1 = start_j1 + 0.2
        pts = []
        for i in range(50):
            f = i / 49.0
            pts.append([start_j1 + (target_j1 - start_j1) * f] + self.current_joints[1:])
        self.traj_points = pts
        self.traj_start_time = rospy.Time.now()
        self.executing = True

    def update_motion(self):
        if not self.executing or not self.traj_points:
            return
        elapsed = (rospy.Time.now() - self.traj_start_time).to_sec()
        if elapsed >= self.traj_duration:
            self.current_joints = self.traj_points[-1]
            self.executing = False
        else:
            idx = int(elapsed / self.traj_duration * (len(self.traj_points) - 1))
            idx = min(idx, len(self.traj_points) - 1)
            self.current_joints = self.traj_points[idx]

    def _publish_js(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
                    'joint7', 'joint8']
        g = min(max(self.width / 2.0, 0.0), 0.035)
        msg.position = list(self.current_joints) + [g, -g]
        msg.velocity = [0.0] * 8
        msg.effort = [0.0] * 8
        self.js_pub.publish(msg)

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.update_motion()
            self._publish_js()
            rate.sleep()

if __name__ == '__main__':
    node = GraspSimNode()
    node.run()