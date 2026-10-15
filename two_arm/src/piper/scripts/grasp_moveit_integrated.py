#!/usr/bin/env python3
"""
集成节点：
1. 订阅 /hggd_pose_raw，选择最佳抓取
2. 使用 MoveIt 规划从当前关节状态到预抓取/抓取/退回的路径
3. 通过 /joint_ctrl_single 将轨迹发送给真实臂
4. 控制夹爪开合
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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive

from piper_msgs.srv import Gripper, GripperRequest
from tf.transformations import quaternion_from_matrix, euler_from_quaternion

class GraspMoveItPipeline:
    def __init__(self):
        # ---------- 初始化 MoveIt ----------
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("arm_group")

        # 配置规划参数
        self.group.set_planning_time(5.0)
        self.group.set_max_velocity_scaling_factor(0.1)      # 低速
        self.group.set_max_acceleration_scaling_factor(0.1)

        # ---------- 夹爪服务 ----------
        rospy.loginfo("Waiting for /gripper_srv ...")
        rospy.wait_for_service('/gripper_srv')
        self.gripper_srv = rospy.ServiceProxy('/gripper_srv', Gripper)

        # ---------- TF ----------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ---------- 真实臂控制接口 ----------
        self.joint_pub = rospy.Publisher("/joint_ctrl_single", JointState, queue_size=1)
        self.pos_pub = rospy.Publisher("/pos_cmd", None, queue_size=1)  # 备选

        # ---------- 碰撞物体：添加桌子 ----------
        self.add_table()

        # ---------- 抓取消息订阅 ----------
        self.best_grasp = None   # 缓存最近一次最佳抓取
        rospy.Subscriber("/hggd_pose_raw", String, self.grasp_cb)

        rospy.loginfo("Grasp Pipeline Ready. Waiting for grasp messages...")

    def add_table(self):
        """添加桌面碰撞物体（尺寸根据实际测量调整）"""
        rospy.sleep(1.0)
        co = CollisionObject()
        co.id = "table"
        co.header.frame_id = "world"   # 需要 world → arm_base 静态 TF 已发布
        co.operation = CollisionObject.ADD

        # 长方体：长(x) 1.0m, 宽(y) 2.0m, 高(z) 0.05m
        primitive = SolidPrimitive(type=1, dimensions=[1.0, 2.0, 0.05])
        co.primitives.append(primitive)

        # 桌面的位姿：中心在 (0.5, 0, -0.025)，上表面在 z=0 平面
        table_pose = geometry_msgs.msg.Pose()
        table_pose.position = Point(x=0.5, y=0.0, z=-0.025)
        table_pose.orientation.w = 1.0
        co.primitive_poses.append(table_pose)

        self.scene.add_object(co)
        rospy.loginfo("Table collision object added.")

    def grasp_cb(self, msg):
        """接收到 HGGD 抓取结果时保存最佳"""
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logerr("Invalid JSON: %s", e)
            return
        if not data:
            return
        self.best_grasp = max(data, key=lambda g: g["score"])
        rospy.loginfo("Updated best grasp (score: %.2f, width: %.3f)",
                      self.best_grasp["score"], self.best_grasp["width"])

    def wait_for_grasp(self, timeout=30.0):
        """阻塞等待直到收到抓取"""
        start = rospy.Time.now()
        while self.best_grasp is None and not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logerr("Timeout waiting for grasp.")
                return False
            rospy.sleep(0.1)
        return True

    def get_target_pose(self):
        """将当前最佳抓取转换为 arm_base 坐标系下的 PoseStamped"""
        g = self.best_grasp
        tx, ty, tz = g["translation"]
        rot = g["rotation"]

        # 旋转矩阵 → 四元数
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

        try:
            transform = self.tf_buffer.lookup_transform(
                "arm_base",
                pose_cam.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            pose_arm = tf2_geometry_msgs.do_transform_pose(pose_cam, transform)
            return pose_arm
        except Exception as e:
            rospy.logerr("TF failed: %s", e)
            return None

    def plan_to_pose(self, target_pose):
        """设置目标位姿并规划"""
        self.group.set_pose_target(target_pose)
        plan = self.group.plan()
        if not plan or not plan.joint_trajectory.points:
            rospy.logerr("Planning failed")
            return None
        rospy.loginfo("Planning succeeded with %d points", len(plan.joint_trajectory.points))
        return plan

    def execute_trajectory(self, plan):
        """将规划出的关节轨迹通过 /joint_ctrl_single 逐点发送"""
        traj = plan.joint_trajectory
        if not traj.points:
            rospy.logerr("Empty trajectory")
            return False

        # 提取关节名称顺序（应匹配真实臂）
        joint_names = traj.joint_names
        # 提取时间间隔
        dt = traj.points[1].time_from_start.to_sec() - traj.points[0].time_from_start.to_sec() \
             if len(traj.points) > 1 else 0.1
        dt = max(dt, 0.05)   # 最小 0.05 秒

        rospy.loginfo("Sending trajectory (%d points, dt=%.2fs)...", len(traj.points), dt)

        for pt in traj.points:
            if rospy.is_shutdown():
                return False
            # 构造 JointState 消息
            js = JointState()
            js.header.stamp = rospy.Time.now()
            js.name = joint_names
            js.position = list(pt.positions)
            js.velocity = list(pt.velocities) if pt.velocities else [0.0]*len(pt.positions)
            js.effort = [0.0]*len(pt.positions)
            self.joint_pub.publish(js)
            rospy.sleep(dt)

        rospy.loginfo("Trajectory execution finished.")
        return True

    def call_gripper(self, opening_m, effort=2.0):
        """控制夹爪开合"""
        try:
            req = GripperRequest()
            req.gripper_angle = opening_m
            req.gripper_effort = effort
            req.set_zero = 0
            req.gripper_code = 1
            resp = self.gripper_srv(req)
            rospy.loginfo("Gripper called: angle=%.3f success=%s", opening_m, resp)
            return True
        except rospy.ServiceException as e:
            rospy.logerr("Gripper service failed: %s", e)
            return False

    def user_confirm(self, prompt="Continue? (y/n): "):
        """用户确认"""
        answer = input(prompt)
        return answer.lower() == 'y'

    def run(self):
        """主流程：等待抓取 → 规划 → 确认 → 执行"""
        rospy.loginfo("Waiting for grasp...")
        if not self.wait_for_grasp():
            return

        # 获取目标位姿
        target_pose = self.get_target_pose()
        if target_pose is None:
            return

        rospy.loginfo("Target pose in arm_base:\n%s", target_pose.pose)

        # 构造预抓取位姿（沿抓取方向后退 0.1m）
        pre_grasp = PoseStamped()
        pre_grasp.header.frame_id = target_pose.header.frame_id
        # 抓取方向：通常是末端 Z 轴负方向？这里取夹爪接近方向（相机光轴方向）
        # 简单起见，沿目标位姿的 Z 轴负方向后退 0.1m
        rot_mat = SR.from_quat([
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w
        ]).as_matrix()
        approach_vector = -rot_mat[:, 2]   # Z 轴负方向（接近方向）
        pre_grasp.pose.position.x = target_pose.pose.position.x + approach_vector[0] * 0.1
        pre_grasp.pose.position.y = target_pose.pose.position.y + approach_vector[1] * 0.1
        pre_grasp.pose.position.z = target_pose.pose.position.z + approach_vector[2] * 0.1
        pre_grasp.pose.orientation = target_pose.pose.orientation

        # 规划预抓取
        rospy.loginfo("Planning to pre-grasp pose...")
        plan_pre = self.plan_to_pose(pre_grasp)
        if plan_pre is None:
            return
        # 规划最终抓取
        rospy.loginfo("Planning to final grasp pose...")
        plan_grasp = self.plan_to_pose(target_pose)
        if plan_grasp is None:
            return

        # 用户确认
        if not self.user_confirm("Execute pre-grasp movement? (y/n): "):
            return

        # 执行预抓取
        self.execute_trajectory(plan_pre)

        # 打开夹爪
        self.call_gripper(0.07, effort=2.0)

        if not self.user_confirm("Move to final grasp? (y/n): "):
            return
        # 执行最终抓取
        self.execute_trajectory(plan_grasp)

        # 关闭夹爪
        width = self.best_grasp["width"] if self.best_grasp else 0.02
        self.call_gripper(max(min(width, 0.07), 0.01), effort=2.0)

        # 退回预抓取位姿
        if self.user_confirm("Return to pre-grasp? (y/n): "):
            self.execute_trajectory(plan_pre)

        rospy.loginfo("Grasp pipeline finished.")

if __name__ == '__main__':
    rospy.init_node("grasp_moveit_integrated")
    import tf2_geometry_msgs
    node = GraspMoveItPipeline()
    node.run()
