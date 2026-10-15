#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grasp_moveit_sim_v2.py

单臂抓取仿真节点 v2：
1. 持续发布 /joint_states_hggd，保证 robot_state_publisher / MoveIt TF 稳定
2. 支持启动后先从下垂姿态移动到观察姿态
3. 可选调用 /trigger_inference 触发 HGGD
4. 接收 /hggd_pose_raw 后，不再只取最高分，而是按 score 排序逐个尝试规划
5. 规划 pre_grasp -> grasp 两段轨迹，第二段从 pre_grasp 的末端状态开始规划，避免仿真跳变
6. 发布完整 8 关节 JointState，避免轨迹执行时夹爪关节从 TF 中消失

注意：
- 这是仿真执行节点，只发布 /joint_states_hggd，不直接控制真实机械臂。
- observation/safe_lift 关节角需要根据实际侧装方向在 RViz 中调整。
"""

import copy
import json
import sys
import threading

import numpy as np
import rospy
import moveit_commander
import tf2_ros
import tf2_geometry_msgs

import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped, Point
from moveit_msgs.msg import CollisionObject, RobotState
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger
from scipy.spatial.transform import Rotation as SR
from tf.transformations import quaternion_from_matrix


class GraspMoveItSimV2:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        # -------------------------
        # Parameters
        # -------------------------
        self.joint_names_6 = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_names_8 = self.joint_names_6 + ['joint7', 'joint8']

        # 模拟自然下垂初始位姿。请用实际仿真确认是否与真实侧装方向一致。
        self.current_joints = rospy.get_param(
            "~initial_joints",
            [0.0, 3.142, -2.967, 0.0, 0.0, 0.0]
        )

        # 两个固定观察姿态只是默认示例，强烈建议在 RViz 里根据侧装方向微调。
        self.safe_lift_joints = rospy.get_param(
            "~safe_lift_joints",
            [0.0, 2.65, -2.35, 0.0, 0.25, 0.0]
        )
        self.look_table_joints = rospy.get_param(
            "~look_table_joints",
            [0.0, 2.10, -1.85, 0.0, 0.65, 0.0]
        )

        # 可选扫描姿态。若 HGGD 第一次无结果，可以手动或自动依次测试。
        self.scan_joints_list = rospy.get_param(
            "~scan_joints_list",
            [
                [0.25, 2.10, -1.85, 0.0, 0.65, 0.0],
                [-0.25, 2.10, -1.85, 0.0, 0.65, 0.0],
            ]
        )

        self.width = float(rospy.get_param("~initial_gripper_width", 0.04))
        self.max_gripper_width = float(rospy.get_param("~max_gripper_width", 0.08))
        self.pregrasp_distance = float(rospy.get_param("~pregrasp_distance", 0.10))
        self.max_candidates = int(rospy.get_param("~max_candidates", 8))

        self.auto_observe = bool(rospy.get_param("~auto_observe", True))
        self.trigger_hggd_after_observe = bool(rospy.get_param("~trigger_hggd_after_observe", False))
        self.auto_scan_if_no_grasp = bool(rospy.get_param("~auto_scan_if_no_grasp", False))
        self.grasp_timeout = float(rospy.get_param("~grasp_timeout", 5.0))

        self.planning_time = float(rospy.get_param("~planning_time", 5.0))
        self.velocity_scale = float(rospy.get_param("~velocity_scale", 0.20))
        self.acceleration_scale = float(rospy.get_param("~acceleration_scale", 0.20))

        # 只用于避障显示/规划。若你已经在别处添加了偏大的 robot body，可将 add_robot_body 设为 false。
        self.add_body = bool(rospy.get_param("~add_robot_body", False))

        # -------------------------
        # Joint state publisher
        # -------------------------
        self.js_pub = rospy.Publisher("/joint_states_hggd", JointState, queue_size=20)
        self.pub_timer = rospy.Timer(rospy.Duration(0.1), self.publish_current_js)

        rospy.sleep(1.0)

        # -------------------------
        # MoveIt setup
        # -------------------------
        self.robot = moveit_commander.RobotCommander()
        rospy.loginfo("Waiting for move_group to be ready...")

        arm_group_name = rospy.get_param("~arm_group", "")
        if not arm_group_name:
            arm_group_name = self.find_arm_group()

        if not arm_group_name:
            rospy.logerr("No arm group found. Exiting.")
            sys.exit(1)

        rospy.loginfo("Using arm group: %s", arm_group_name)
        self.group = self.wait_for_group(arm_group_name)

        self.planning_frame = self.group.get_planning_frame()
        rospy.loginfo("Planning frame: %s", self.planning_frame)

        self.group.set_planning_time(self.planning_time)
        self.group.set_max_velocity_scaling_factor(self.velocity_scale)
        self.group.set_max_acceleration_scaling_factor(self.acceleration_scale)
        self.group.allow_replanning(True)

        # 可选：如果 URDF/SRDF 中已经定义了真正 TCP，可在 launch 中传入 ~eef_link。
        eef_link = rospy.get_param("~eef_link", "")
        if eef_link:
            try:
                self.group.set_end_effector_link(eef_link)
                rospy.loginfo("Using end-effector link: %s", self.group.get_end_effector_link())
            except Exception as e:
                rospy.logwarn("Failed to set eef_link=%s: %s", eef_link, e)

        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(1.0)
        if self.add_body:
            self.add_robot_body()

        # -------------------------
        # TF and grasp subscription
        # -------------------------
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.last_grasp_msg_time = None
        self.last_grasps = []
        self.busy = False
        self.lock = threading.Lock()

        rospy.Subscriber("/hggd_pose_raw", String, self.grasp_cb, queue_size=1)

        rospy.loginfo("grasp_moveit_sim_v2 ready.")

        if self.auto_observe:
            rospy.Timer(rospy.Duration(1.0), self.start_observation_once, oneshot=True)

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def find_arm_group(self):
        for _ in range(60):
            groups = self.robot.get_group_names()
            rospy.loginfo("Current MoveIt groups: %s", groups)
            for name in groups:
                if "arm" in name.lower():
                    return name
            rospy.sleep(1.0)
        return None

    def wait_for_group(self, group_name):
        for _ in range(30):
            try:
                return moveit_commander.MoveGroupCommander(group_name)
            except RuntimeError:
                rospy.sleep(1.0)
        rospy.logerr("Failed to connect to move_group.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Joint state publishing
    # ------------------------------------------------------------------
    def publish_current_js(self, event=None):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names_8

        g = self.width / 2.0
        msg.position = list(self.current_joints[:6]) + [g, -g]
        msg.velocity = [0.0] * 8
        msg.effort = [0.0] * 8
        self.js_pub.publish(msg)

    def set_current_joints_from_trajectory_point(self, joint_names, positions):
        name_to_pos = dict(zip(joint_names, positions))
        for i, name in enumerate(self.joint_names_6):
            if name in name_to_pos:
                self.current_joints[i] = name_to_pos[name]

    def make_robot_state_from_joints(self, joint_names, positions):
        state = RobotState()
        state.joint_state.header.stamp = rospy.Time.now()
        state.joint_state.name = list(joint_names)
        state.joint_state.position = list(positions)
        return state

    # ------------------------------------------------------------------
    # Collision object
    # ------------------------------------------------------------------
    def add_robot_body(self):
        co = CollisionObject()
        co.id = "robot_body_oversized"
        co.header.frame_id = self.planning_frame
        co.operation = CollisionObject.ADD

        # 默认偏大模型，仅兜底。你已有更标准/更偏大的模型时建议关掉本函数。
        body = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.2, 0.6, 0.9])
        co.primitives.append(body)

        body_pose = geometry_msgs.msg.Pose()
        body_pose.position = Point(x=0.35, y=0.0, z=-0.45)
        body_pose.orientation.w = 1.0
        co.primitive_poses.append(body_pose)

        self.scene.add_object(co)
        rospy.loginfo("Oversized robot body collision object added.")

    # ------------------------------------------------------------------
    # Observation stage
    # ------------------------------------------------------------------
    def start_observation_once(self, event=None):
        rospy.loginfo("Auto observation enabled: moving from hanging pose to observation pose.")
        ok = self.move_to_joint_pose(self.safe_lift_joints, label="safe_lift")
        if not ok:
            rospy.logerr("Failed to reach safe_lift_joints. Stop observation.")
            return

        ok = self.move_to_joint_pose(self.look_table_joints, label="look_table")
        if not ok:
            rospy.logerr("Failed to reach look_table_joints. Stop observation.")
            return

        rospy.loginfo("Observation pose reached. Camera should now be able to inspect the table if pose is tuned.")

        if self.trigger_hggd_after_observe:
            self.trigger_hggd_and_optionally_scan()

    def trigger_hggd_and_optionally_scan(self):
        self.trigger_hggd()
        if self.wait_for_grasp(self.grasp_timeout):
            rospy.loginfo("HGGD grasp received after look_table pose.")
            return

        rospy.logwarn("No HGGD grasp received after %.1fs.", self.grasp_timeout)

        if not self.auto_scan_if_no_grasp:
            return

        for idx, scan_pose in enumerate(self.scan_joints_list):
            ok = self.move_to_joint_pose(scan_pose, label="scan_%d" % idx)
            if not ok:
                continue
            self.trigger_hggd()
            if self.wait_for_grasp(self.grasp_timeout):
                rospy.loginfo("HGGD grasp received at scan_%d.", idx)
                return

        rospy.logwarn("No HGGD grasp received after scan poses.")

    def trigger_hggd(self):
        try:
            rospy.wait_for_service("/trigger_inference", timeout=2.0)
            trigger = rospy.ServiceProxy("/trigger_inference", Trigger)
            resp = trigger()
            rospy.loginfo("trigger_inference response: success=%s message=%s", resp.success, resp.message)
        except Exception as e:
            rospy.logwarn("Failed to call /trigger_inference. If HGGD is not running, publish a test grasp manually. Error: %s", e)

    def wait_for_grasp(self, timeout):
        start = rospy.Time.now()
        old_time = self.last_grasp_msg_time
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < timeout:
            if self.last_grasp_msg_time is not None and self.last_grasp_msg_time != old_time:
                return True
            rate.sleep()
        return False

    # ------------------------------------------------------------------
    # Planning: joint poses
    # ------------------------------------------------------------------
    def move_to_joint_pose(self, joints, label="joint_pose"):
        if len(joints) != 6:
            rospy.logerr("%s requires 6 joint values, got %d", label, len(joints))
            return False

        rospy.loginfo("Planning to %s: %s", label, ["%.3f" % v for v in joints])

        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(dict(zip(self.joint_names_6, joints)))

        plan = self.extract_plan(self.group.plan())
        if plan is None:
            rospy.logerr("Planning to %s failed.", label)
            return False

        rospy.loginfo("Executing simulated trajectory to %s.", label)
        self.publish_trajectory(plan)
        return True

    # ------------------------------------------------------------------
    # HGGD callback and grasp candidate selection
    # ------------------------------------------------------------------
    def grasp_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logerr("Invalid HGGD JSON: %s", e)
            return

        if not isinstance(data, list) or len(data) == 0:
            rospy.logwarn("HGGD message contains no grasp candidates.")
            return

        self.last_grasp_msg_time = rospy.Time.now()
        self.last_grasps = data

        with self.lock:
            if self.busy:
                rospy.logwarn("Already planning/executing. Ignore this HGGD message.")
                return
            self.busy = True

        try:
            self.plan_and_execute_best_candidate(data)
        finally:
            with self.lock:
                self.busy = False

    def plan_and_execute_best_candidate(self, grasps):
        valid = []
        for g in grasps:
            try:
                score = float(g.get("score", 0.0))
                width = float(g.get("width", self.width))
                if width > self.max_gripper_width:
                    rospy.logwarn(
                        "Skip grasp score=%.3f because width %.3f > max %.3f",
                        score, width, self.max_gripper_width
                    )
                    continue
                valid.append(g)
            except Exception:
                continue

        if not valid:
            rospy.logerr("No valid grasp candidate after width filtering.")
            return

        valid = sorted(valid, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        valid = valid[:self.max_candidates]

        rospy.loginfo("Trying %d grasp candidates by score.", len(valid))

        for idx, grasp in enumerate(valid):
            score = float(grasp.get("score", 0.0))
            rospy.loginfo("Candidate %d/%d score=%.3f", idx + 1, len(valid), score)

            target_pose = self.grasp_to_pose_in_planning_frame(grasp)
            if target_pose is None:
                continue

            pre_grasp = self.make_pregrasp_pose(target_pose)
            if pre_grasp is None:
                continue

            # 规划 1：当前位置 -> pre_grasp
            self.group.clear_pose_targets()
            self.group.set_start_state_to_current_state()
            plan_pre = self.plan_to_pose(pre_grasp, label="pre_grasp")
            if plan_pre is None:
                continue

            # 规划 2：pre_grasp 末端状态 -> grasp
            last = plan_pre.joint_trajectory.points[-1]
            start_state = self.make_robot_state_from_joints(
                plan_pre.joint_trajectory.joint_names,
                last.positions
            )
            self.group.clear_pose_targets()
            self.group.set_start_state(start_state)
            plan_grasp = self.plan_to_pose(target_pose, label="grasp")
            if plan_grasp is None:
                continue

            rospy.loginfo("Candidate %d accepted. Simulating pre_grasp -> grasp.", idx + 1)
            self.publish_trajectory(plan_pre)
            rospy.sleep(0.3)
            self.publish_trajectory(plan_grasp)

            self.width = min(float(grasp.get("width", self.width)), self.max_gripper_width)
            self.publish_current_js()
            return

        rospy.logerr("All grasp candidates failed planning.")

    # ------------------------------------------------------------------
    # Grasp pose conversion
    # ------------------------------------------------------------------
    def grasp_to_pose_in_planning_frame(self, grasp):
        try:
            tx, ty, tz = grasp["translation"]
            rot = grasp["rotation"]
            quat = quaternion_from_matrix([
                [rot[0][0], rot[0][1], rot[0][2], 0],
                [rot[1][0], rot[1][1], rot[1][2], 0],
                [rot[2][0], rot[2][1], rot[2][2], 0],
                [0, 0, 0, 1]
            ])
        except Exception as e:
            rospy.logwarn("Bad grasp candidate format: %s", e)
            return None

        pose_cam = PoseStamped()
        pose_cam.header.frame_id = "camera_color_optical_frame"
        pose_cam.header.stamp = rospy.Time(0)
        pose_cam.pose.position.x = tx
        pose_cam.pose.position.y = ty
        pose_cam.pose.position.z = tz
        pose_cam.pose.orientation.x = quat[0]
        pose_cam.pose.orientation.y = quat[1]
        pose_cam.pose.orientation.z = quat[2]
        pose_cam.pose.orientation.w = quat[3]

        for attempt in range(10):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.planning_frame,
                    pose_cam.header.frame_id,
                    rospy.Time(0),
                    rospy.Duration(1.0)
                )
                pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, transform)
                pose_base.header.frame_id = self.planning_frame
                pose_base.header.stamp = rospy.Time.now()
                rospy.loginfo(
                    "Target in %s: xyz=(%.3f, %.3f, %.3f)",
                    self.planning_frame,
                    pose_base.pose.position.x,
                    pose_base.pose.position.y,
                    pose_base.pose.position.z
                )
                return pose_base
            except Exception as e:
                if attempt < 9:
                    rospy.sleep(0.3)
                else:
                    rospy.logerr("TF transform failed after 10 attempts: %s", e)
                    return None

    def make_pregrasp_pose(self, target_pose):
        pre = copy.deepcopy(target_pose)

        try:
            rot_mat = SR.from_quat([
                target_pose.pose.orientation.x,
                target_pose.pose.orientation.y,
                target_pose.pose.orientation.z,
                target_pose.pose.orientation.w
            ]).as_matrix()
        except Exception as e:
            rospy.logwarn("Invalid target quaternion: %s", e)
            return None

        # 默认沿抓取坐标系 -Z 后退。若 HGGD 坐标定义不同，只需要改这里。
        approach = -rot_mat[:, 2]
        pre.pose.position.x += approach[0] * self.pregrasp_distance
        pre.pose.position.y += approach[1] * self.pregrasp_distance
        pre.pose.position.z += approach[2] * self.pregrasp_distance

        rospy.loginfo(
            "Pre-grasp in %s: xyz=(%.3f, %.3f, %.3f)",
            pre.header.frame_id,
            pre.pose.position.x,
            pre.pose.position.y,
            pre.pose.position.z
        )
        return pre

    # ------------------------------------------------------------------
    # Planning: pose target
    # ------------------------------------------------------------------
    def plan_to_pose(self, target_pose, label="pose"):
        self.group.clear_pose_targets()
        self.group.set_pose_target(target_pose)

        plan = self.extract_plan(self.group.plan())
        self.group.clear_pose_targets()

        if plan is None:
            rospy.logwarn("Planning to %s failed.", label)
            return None

        rospy.loginfo(
            "Planning to %s succeeded with %d points.",
            label,
            len(plan.joint_trajectory.points)
        )
        return plan

    def extract_plan(self, plan_result):
        # MoveIt Noetic Python 常见返回：
        # - RobotTrajectory
        # - tuple(success, trajectory, planning_time, error_code)
        if isinstance(plan_result, tuple):
            if len(plan_result) >= 2:
                success = plan_result[0]
                traj = plan_result[1]
                if not success:
                    return None
                plan_result = traj
            else:
                return None

        if not hasattr(plan_result, "joint_trajectory"):
            return None

        if not plan_result.joint_trajectory.points:
            return None

        return plan_result

    # ------------------------------------------------------------------
    # Simulated trajectory execution
    # ------------------------------------------------------------------
    def publish_trajectory(self, plan):
        traj = plan.joint_trajectory
        if not traj.points:
            return

        joint_names = traj.joint_names

        last_t = 0.0
        for pt in traj.points:
            if rospy.is_shutdown():
                return

            self.set_current_joints_from_trajectory_point(joint_names, pt.positions)
            self.publish_current_js()

            t = pt.time_from_start.to_sec()
            dt = max(t - last_t, 0.03)
            last_t = t
            rospy.sleep(dt)

        self.publish_current_js()


if __name__ == "__main__":
    rospy.init_node("grasp_moveit_sim_v2")
    node = GraspMoveItSimV2()
    rospy.spin()
