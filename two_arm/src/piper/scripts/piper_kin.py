#!/usr/bin/env python3
""" piper 运动学共享模块 (无ROS依赖, 供 A/B 运动规划器复用)

    与 ik_check.py 同源的模型构建:
      - URDF锁定 joint7/8(夹爪), ee帧挂joint6(恒等, =SDK笛卡尔系, 已实测恒等)
      - 自碰撞几何对配置与 Arm_IK 相同
    提供: FK / LOCAL雅可比 / 自碰撞检查
"""

import os

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

CUR = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(os.path.dirname(os.path.dirname(CUR)),
                    'piper_description/urdf/piper_description.urdf')


class PiperKin:
    def __init__(self, urdf=URDF):
        
        # 运动学模型，加载URDF
        robot = RobotWrapper.BuildFromURDF(urdf)
        # 利用URDF建立6维降维模型 model（锁夹爪关节, 只留6关节，降低自由度）
        self.model = robot.buildReducedRobot(
            list_of_joints_to_lock=["joint7", "joint8"],
            reference_configuration=np.array([0] * robot.model.nq),
        ).model
        # 添加运动学FrameType.OP_FRAME可执行运动帧ee帧方便后续运动学调用，挂在joint6处，两者Quaternion(1, 0, 0, 0)恒等旋转且np.zeros(3)零平移量
        self.model.addFrame(pin.Frame(
            'ee', self.model.getJointId('joint6'),
            pin.SE3(pin.Quaternion(1, 0, 0, 0), np.zeros(3)),
            pin.FrameType.OP_FRAME))
        # data工作区为静态模型model的动态结果缓存区
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId('ee')
        self.nq = self.model.nq

        # 自碰撞模型，构建全模型(含夹爪)的自碰撞几何模型full_model，与18对碰撞对模型geom_model：远端组（前臂+手腕+夹爪两指）× 近端组（基座+大臂+小臂），共 6×3=18 对，data为对应动态结果缓存
        self.full_model = robot.model
        self.geom_model = pin.buildGeomFromUrdf(
            robot.model, urdf, pin.GeometryType.COLLISION)
        for i in range(4, 10):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geom_data = pin.GeometryData(self.geom_model)
        # 全模型配套data(FK/几何摆放共用一个)缓存复用
        self.full_data = self.full_model.createData()

        # 关节限位(与URDF一致)
        self.q_lo = self.model.lowerPositionLimit.copy()
        self.q_hi = self.model.upperPositionLimit.copy()

    # 正运动学，关节组q -> SE3末端位姿
    def fk(self, q) -> pin.SE3:
        """末端(link6)位姿"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_id)
        return self.data.oMf[self.ee_id]

    # 雅阁比，当前关节组q的雅各比矩阵
    def jac_local(self, q) -> np.ndarray:
        """LOCAL(body)系 6x6 雅可比 —— 与 log6 误差配套使用"""
        return pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_id, pin.LOCAL)

    # 当前关节组q自碰撞检测
    def self_collision(self, q, gripper=np.zeros(2)) -> bool:
        """全模型(含夹爪)自碰撞检查"""
        q_full = np.concatenate([q, gripper])
        pin.forwardKinematics(self.full_model, self.full_data, q_full)
        pin.updateGeometryPlacements(self.full_model, self.full_data,
                                     self.geom_model, self.geom_data)
        return pin.computeCollisions(self.geom_model, self.geom_data, False)


def se3_from_xyzrpy(xyz, rpy) -> pin.SE3:
    """平移 + rpy(弧度, sxyz与tf一致) -> SE3"""
    from scipy.spatial.transform import Rotation as SR
    return pin.SE3(SR.from_euler('xyz', rpy).as_matrix(), np.array(xyz))
