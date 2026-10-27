#!/usr/bin/env python3
""" A规划器(LineQPPlanner)真机直线段测试脚本

    场景: 双臂launch已在跑(2arm_camera_rviz.launch, 提供joint反馈/joint_ctrl_single/使能服务),
          本脚本在两个工作空间位姿之间用A规划器执行一小段严格直线的平移。

    流程(三段, 每段执行前回车确认):
        ① 当前位形 -> P1 (就位)
        ② P1 -> P2   (测试主体: 两点同姿态时为纯平移直线)
        ③ P2 -> P1   (原路返回)

    执行方式: 规划出的关节航点流按 dts 节奏逐点发 /{arm_ns}/joint_ctrl_single,
              固件按关节模式逐点跟踪(MotionCtrl速度参数由 velocity[6] 设定)。
              发送是fire-and-forget, 不等待固件到位, 靠航点间距控制节奏。

    运行环境(主环境即可, pinocchio/casadi已按requirements-planning.txt装好):
        rosrun piper line_test_real.py _arm_ns:=arm1 \
            _p1:="[0.30,0.0,0.25,0.0,1.4838,0.0]" \
            _p2:="[0.30,0.0,0.32,0.0,1.4838,0.0]" \
            _v_lin:=0.05 _qd_max:=1.0 _move_speed:=30

    参数:
        ~arm_ns     臂命名空间, 默认arm1
        ~p1/~p2     两点的位姿 [x y z roll pitch yaw] (米/弧度, 臂基系, 工具对齐语义)
        ~v_lin      末端线速度上限 m/s (默认0.05, 首跑建议0.03)
        ~a_lin      末端线加速度 m/s^2 (默认0.20)
        ~qd_max     关节速度上限 rad/s (默认1.0, 首跑建议0.5)
        ~move_speed 固件运动速度参数0-100 (velocity[6], 默认30)
        ~gripper    全程夹爪开度(米, 0-0.08, 默认0.0闭合)

    安全: 每段只回车才执行; Ctrl-C停止发送(臂停在最后一个已发微航点);
          首跑请把 v_lin/qd_max/move_speed 都减半, 臂周围清空。
"""

import sys

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from piper_kin import PiperKin, se3_from_xyzrpy
from motion_a_line_qp import LineQPPlanner


def get_current_joints(arm_ns, timeout=5.0):
    """读当前6关节(rad), 顺带验证SDK节点在发反馈"""
    msg = rospy.wait_for_message("/%s/joint_states_single" % arm_ns,
                                 JointState, timeout=timeout)
    return np.array(msg.position[:6], dtype=float)


def stream_joints(pub, traj, dts, gripper, move_speed):
    """按规划节奏逐点发关节流(位置rad+夹爪m, velocity[6]=固件速度参数)"""
    for q, dt in zip(traj[1:], dts):
        m = JointState()
        m.header.stamp = rospy.Time.now()
        m.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
        m.position = list(q) + [gripper]
        m.velocity = [0.0] * 6 + [float(move_speed)]
        m.effort = [0.0] * 6 + [0.5]
        pub.publish(m)
        rospy.sleep(dt)


def confirm(reminder):
    a = input("[0退出] " + reminder)
    if a == "0":
        rospy.signal_shutdown("user abort")
        sys.exit(0)


def plan_and_run(planner, kin, arm_ns, pub, T_goal, gripper, move_speed, label):
    """读当前关节 -> 规划 -> 打印体检 -> 确认 -> 流式执行"""
    q_now = get_current_joints(arm_ns)
    T0 = kin.fk(q_now)
    rospy.loginfo("[%s] 当前末端 xyz=%s -> 目标 xyz=%s",
                  label, np.round(T0.translation, 4), np.round(T_goal.translation, 4))
    traj, dts, stats = planner.plan(q_now, T_goal)
    rospy.loginfo("[%s] 航点%d 时长%.2fs 直线度%.5fm 终点误差 pos%.4fm ori%.4frad",
                  label, stats["steps"], stats["duration"],
                  stats["max_line_deviation"],
                  stats["final_pos_err"], stats["final_ori_err"])
    confirm("%s 回车执行: " % label)
    stream_joints(pub, traj, dts, gripper, move_speed)
    rospy.sleep(0.5)


def main():
    rospy.init_node("line_test_real", anonymous=True)
    arm_ns = rospy.get_param("~arm_ns", "arm1")
    p1 = [float(v) for v in rospy.get_param("~p1", [0.30, 0.0, 0.25, 0.0, 1.4838, 0.0])]
    p2 = [float(v) for v in rospy.get_param("~p2", [0.30, 0.0, 0.32, 0.0, 1.4838, 0.0])]
    v_lin = float(rospy.get_param("~v_lin", 0.05))
    a_lin = float(rospy.get_param("~a_lin", 0.20))
    qd_max = float(rospy.get_param("~qd_max", 1.0))
    move_speed = int(rospy.get_param("~move_speed", 30))
    gripper = float(rospy.get_param("~gripper", 0.0))

    kin = PiperKin()
    planner = LineQPPlanner(kin, v_lin=v_lin, a_lin=a_lin, qd_max=qd_max)
    pub = rospy.Publisher("/%s/joint_ctrl_single" % arm_ns, JointState, queue_size=1)

    rospy.loginfo("臂=%s  P1=%s  P2=%s  v=%.2f qd=%.1f 固件速度=%d 夹爪=%.3f",
                  arm_ns, p1, p2, v_lin, qd_max, move_speed, gripper)
    confirm("参数确认(依赖launch的auto_enable已使能), 回车开始 ①就位段: ")

    T1 = se3_from_xyzrpy(p1[:3], p1[3:])
    T2 = se3_from_xyzrpy(p2[:3], p2[3:])

    plan_and_run(planner, kin, arm_ns, pub, T1, gripper, move_speed, "①就位")
    plan_and_run(planner, kin, arm_ns, pub, T2, gripper, move_speed, "②平移")
    plan_and_run(planner, kin, arm_ns, pub, T1, gripper, move_speed, "③返回")
    rospy.loginfo("三段全部完成")


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as ex:
        rospy.logerr("中止: %s", ex)
        sys.exit(1)
