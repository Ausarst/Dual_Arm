#!/usr/bin/env python3
"""
仿真集成节点（持续关节状态发布版）
- 一直以 10Hz 发布当前关节状态，保证 TF 树始终存在
- 接收 HGGD 抓取位姿，通过 MoveIt 规划并发布轨迹
"""
import rospy
import sys
import json
import numpy as np
from scipy.spatial.transform import Rotation as SR
import moveit_commander
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
import tf2_geometry_msgs
from tf.transformations import quaternion_from_matrix

class GraspMoveItSim:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        # 当前关节状态（初始零位）
        # 模拟竖直下垂完全伸直姿态：关节2到上限，关节3到下限，其余为0
        self.current_joints = [0.0, 3.142, -2.967, 0.0, 0.0, 0.0]
        self.width = 0.04

        # 建立发布者，并用定时器持续发布关节状态（10Hz）
        self.js_pub = rospy.Publisher("/joint_states_hggd", JointState, queue_size=10)
        self.pub_timer = rospy.Timer(rospy.Duration(0.1), self.publish_current_js)

        # 等待 TF 树短暂建立
        rospy.sleep(1.0)

        # 连接 MoveIt
        self.robot = moveit_commander.RobotCommander()
        rospy.loginfo("Waiting for move_group to be ready...")
        arm_group_name = None
        for i in range(60):
            all_groups = self.robot.get_group_names()
            rospy.loginfo("Current groups: %s", all_groups)
            for name in all_groups:
                if 'arm' in name.lower():
                    arm_group_name = name
                    break
            if arm_group_name:
                break
            rospy.sleep(1.0)
        if arm_group_name is None:
            rospy.logerr("No arm group found. Exiting.")
            sys.exit(1)

        rospy.loginfo("Using arm group: %s", arm_group_name)
        for i in range(30):
            try:
                self.group = moveit_commander.MoveGroupCommander(arm_group_name)
                break
            except RuntimeError:
                rospy.sleep(1.0)
        else:
            rospy.logerr("Failed to connect to move_group.")
            sys.exit(1)

        self.planning_frame = self.group.get_planning_frame()
        rospy.loginfo("Planning frame: %s", self.planning_frame)

        self.group.set_planning_time(5.0)
        self.group.set_max_velocity_scaling_factor(0.2)
        self.group.set_max_acceleration_scaling_factor(0.2)

        # 添加桌面碰撞物体
        self.scene = moveit_commander.PlanningSceneInterface()
        self.add_robot_body()       

        # TF 监听
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 订阅 HGGD 抓取位姿
        self.best_grasp = None
        rospy.Subscriber("/hggd_pose_raw", String, self.grasp_cb)

        rospy.loginfo("Simulation grasp node ready. Waiting for HGGD messages...")

    def publish_current_js(self, event=None):
        """持续发布当前关节状态，保证 TF 树始终存活"""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
                    'joint7', 'joint8']
        g = self.width / 2.0
        msg.position = list(self.current_joints) + [g, -g]
        msg.velocity = [0.0]*8
        msg.effort = [0.0]*8
        self.js_pub.publish(msg)

    
    def add_robot_body(self):
        co = CollisionObject()
        co.id = "add_robot_body_front"
        co.header.frame_id = self.planning_frame
        co.operation = CollisionObject.ADD
        body = SolidPrimitive(type=1, dimensions=[1.2, 0.4, 0.8])
        co.primitives.append(body)

        body_pose = geometry_msgs.msg.Pose()
        body_pose.position = Point(x=0.4, y=0.0, z=-0.4)
        body_pose.orientation.w = 1.0
        co.primitive_poses.append(body_pose)

        self.scene.add_object(co)
        rospy.loginfo("Robot body added.")

    def grasp_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logerr("Invalid JSON: %s", e)
            return
        if not data:
            return
        self.best_grasp = max(data, key=lambda g: g["score"])
        rospy.loginfo("Received grasp (score: %.2f)", self.best_grasp["score"])
        self.plan_and_execute()

    def get_target_pose(self):
        g = self.best_grasp
        tx, ty, tz = g["translation"]
        rot = g["rotation"]
        quat = quaternion_from_matrix([
            [rot[0][0], rot[0][1], rot[0][2], 0],
            [rot[1][0], rot[1][1], rot[1][2], 0],
            [rot[2][0], rot[2][1], rot[2][2], 0],
            [0, 0, 0, 1]
        ])
        pose_cam = PoseStamped()
        pose_cam.header.frame_id = "camera_color_optical_frame"
        pose_cam.header.stamp = rospy.Time.now()
        pose_cam.pose.position.x = tx
        pose_cam.pose.position.y = ty
        pose_cam.pose.position.z = tz
        pose_cam.pose.orientation.x = quat[0]
        pose_cam.pose.orientation.y = quat[1]
        pose_cam.pose.orientation.z = quat[2]
        pose_cam.pose.orientation.w = quat[3]

        # 等待 TF 树连通，重试 10 次
        for attempt in range(10):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.planning_frame,
                    pose_cam.header.frame_id,
                    rospy.Time(0),
                    rospy.Duration(1.0)
                )
                pose_arm = tf2_geometry_msgs.do_transform_pose(pose_cam, transform)
                return pose_arm
            except Exception as e:
                if attempt < 9:
                    rospy.sleep(0.5)
                else:
                    rospy.logerr("TF failed after 10 attempts: %s", e)
                    return None

    def plan_to_pose(self, target_pose):
        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(target_pose)
        plan = self.group.plan()
        # 兼容 MoveIt 不同版本的返回值
        if isinstance(plan, tuple):
            # 常见返回格式：(success, plan, planning_time)
            if len(plan) >= 2 and hasattr(plan[1], 'joint_trajectory'):
                plan = plan[1]
            else:
                rospy.logerr("Planning returned unexpected tuple: %s", plan)
                return None
        if not hasattr(plan, 'joint_trajectory') or not plan.joint_trajectory.points:
            rospy.logerr("Planning failed: no joint trajectory")
            return None
        rospy.loginfo("Planning succeeded with %d points", len(plan.joint_trajectory.points))
        return plan

    def plan_and_execute(self):
        target_pose = self.get_target_pose()
        if target_pose is None:
            return
        rospy.loginfo("Target pose in %s:\n%s", self.planning_frame, target_pose.pose)

        # 预抓取位姿（沿接近方向后退0.1m）
        pre_grasp = PoseStamped()
        pre_grasp.header.frame_id = target_pose.header.frame_id
        rot_mat = SR.from_quat([
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w
        ]).as_matrix()
        approach = -rot_mat[:, 2]   # 假设 Z 轴负向为接近方向
        pre_grasp.pose.position.x = target_pose.pose.position.x + approach[0] * 0.1
        pre_grasp.pose.position.y = target_pose.pose.position.y + approach[1] * 0.1
        pre_grasp.pose.position.z = target_pose.pose.position.z + approach[2] * 0.1
        pre_grasp.pose.orientation = target_pose.pose.orientation

        plan_pre = self.plan_to_pose(pre_grasp)
        if plan_pre is None: return
        plan_grasp = self.plan_to_pose(target_pose)
        if plan_grasp is None: return

        self.publish_trajectory(plan_pre)
        rospy.sleep(0.5)
        self.publish_trajectory(plan_grasp)

    def publish_trajectory(self, plan):
        """将规划轨迹逐点发布，并更新当前关节值"""
        traj = plan.joint_trajectory
        if not traj.points:
            return
        joint_names = traj.joint_names
        dt = 0.1
        if len(traj.points) > 1:
            dt = max(traj.points[1].time_from_start.to_sec() - traj.points[0].time_from_start.to_sec(), 0.05)
        for pt in traj.points:
            if rospy.is_shutdown():
                return
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = joint_names
            msg.position = list(pt.positions)
            msg.velocity = list(pt.velocities) if pt.velocities else [0.0]*len(pt.positions)
            msg.effort = [0.0]*len(pt.positions)
            self.js_pub.publish(msg)
            # 更新当前关节状态（确保定时器也使用最新值）
            for i, name in enumerate(joint_names):
                if i < 6:
                    self.current_joints[i] = pt.positions[i]
            rospy.sleep(dt)

if __name__ == '__main__':
    rospy.init_node("grasp_moveit_sim")
    node = GraspMoveItSim()
    rospy.spin()