#!/usr/bin/env python3
""" Fake piper SDK node for dry-run test (no hardware needed)

    Mimic the software interface of piper_ctrl_single_node for one arm, plus a
    mini executor so rviz actually SHOWS each commanded motion:
        Publish   /{arm_ns}/joint_states_single   joints at 20Hz, holds the LAST
                   commanded pose (pos_cmd -> IK -> joint lerp), gripper opening
                   from gripper_srv is reflected on joint7/8
        Service   /{arm_ns}/gripper_srv           log request, update joint7/8, return success
        Subscribe /{arm_ns}/pos_cmd               cartesian cmd (m / rad) -> numeric IK
                   (SDK DH forward kinematics piper_fk.CalFK + scipy least_squares),
                   success -> move displayed arm to the solution, fail -> warn & keep

    Usage:
        rosrun piper fake_piper_node.py _arm_ns:=arm1
        (normally launched by 2arm_dryrun.launch, no need to run by hand)
"""

import os
import sys

import numpy as np
import rospy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SR
from sensor_msgs.msg import JointState

from piper_msgs.srv import Gripper, GripperResponse
from piper_msgs.msg import PosCmd

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(CUR)), 'piper_sdk', 'piper_sdk', 'kinematics'))
from piper_fk import C_PiperForwardKinematics  # noqa: E402

# joint limits from piper URDF (rad), same as ik_check.py
J_LO = np.deg2rad([-150.0, 0.0, -170.0, -100.0, -69.9, -120.0])
J_HI = np.deg2rad([150.0, 179.9, 0.0, 100.0, 69.9, 120.0])
POS_TOL = 0.005   # m
ORI_TOL = 0.05    # rad
MOVE_TIME = 0.5   # lerp duration for display


class FakePiper:
    def __init__(self):
        rospy.init_node("fake_piper", anonymous=True)
        self.arm_ns = rospy.get_param("~arm_ns", "arm1")

        self.joint_pub = rospy.Publisher(
            "/%s/joint_states_single" % self.arm_ns, JointState, queue_size=1)
        self.joint_states = JointState()
        self.joint_states.name = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
            'joint7', 'joint8']
        self.joint_states.position = [0.0] * 8
        self.joint_states.velocity = [0.0] * 8
        self.joint_states.effort = [0.0] * 8

        self.q_now = np.zeros(6)        # displayed joints
        self.q_start = np.zeros(6)      # lerp start
        self.q_target = np.zeros(6)     # lerp target (last commanded solution)
        self.move_t0 = None             # rospy time when lerp started
        self.fk = C_PiperForwardKinematics()

        rospy.Subscriber("/%s/pos_cmd" % self.arm_ns, PosCmd,
                         self.pos_callback, queue_size=1, tcp_nodelay=True)
        rospy.Service("/%s/gripper_srv" % self.arm_ns, Gripper,
                      self.handle_gripper_service)

        self.pub_timer = rospy.Timer(rospy.Duration(0.05), self.publish_joints)
        rospy.loginfo("fake piper node ready for arm %s", self.arm_ns)

    ### FK / IK on the SDK DH chain
    def fk_pose(self, q):
        """end pose [x,y,z] m + rpy rad (extrinsic xyz), same convention as PosCmd"""
        x, y, z, r, p, yw = self.fk.CalFK(list(q))[5]
        return np.array([x, y, z]) / 1000.0, np.deg2rad([r, p, yw])

    def solve_ik(self, pos, rpy, seeds):
        """numeric IK, returns (q, pos_err, ori_err) of the best attempt or None"""
        R_target = SR.from_euler('xyz', rpy)

        def residual(q):
            p_fk, r_fk = self.fk_pose(q)
            ori = SR.from_euler('xyz', r_fk).inv() * R_target
            return np.concatenate([p_fk - pos, 0.1 * ori.as_rotvec()])

        best = None
        for seed in seeds:
            try:
                sol = least_squares(residual, seed, bounds=(J_LO, J_HI),
                                    xtol=1e-8, ftol=1e-8, max_nfev=400)
            except Exception:
                continue
            p_fk, r_fk = self.fk_pose(sol.x)
            pos_err = np.linalg.norm(p_fk - pos)
            ori_err = np.linalg.norm(
                (SR.from_euler('xyz', r_fk).inv() * R_target).as_rotvec())
            if best is None or pos_err + 0.1 * ori_err < best[0]:
                best = (pos_err + 0.1 * ori_err, sol.x, pos_err, ori_err)
        return best[1:] if best is not None else None

    ### command inputs
    def pos_callback(self, pos_data):
        target_pos = np.array([pos_data.x, pos_data.y, pos_data.z])
        target_rpy = np.array([pos_data.roll, pos_data.pitch, pos_data.yaw])
        seeds = [self.q_target, np.zeros(6),
                 np.random.default_rng().uniform(J_LO, J_HI)]
        result = self.solve_ik(target_pos, target_rpy, seeds)
        if result is None:
            rospy.logwarn("[%s fake] IK solver failed for pos_cmd", self.arm_ns)
            return
        q_sol, pos_err, ori_err = result
        if pos_err > POS_TOL or ori_err > ORI_TOL:
            rospy.logwarn("[%s fake] pos_cmd unreachable (pos_err=%.4f ori_err=%.4f), keep current pose",
                          self.arm_ns, pos_err, ori_err)
            return
        self.q_start = self.q_now.copy()
        self.q_target = q_sol
        self.move_t0 = rospy.Time.now()
        rospy.loginfo("[%s fake] received PosCmd: x=%.4f y=%.4f z=%.4f r=%.4f p=%.4f yaw=%.4f"
                      " -> q(deg)=%s",
                      self.arm_ns, pos_data.x, pos_data.y, pos_data.z,
                      pos_data.roll, pos_data.pitch, pos_data.yaw,
                      np.round(np.degrees(q_sol), 1).tolist())

    def handle_gripper_service(self, req):
        rospy.loginfo("[%s fake] gripper_srv called: angle=%.4f effort=%.2f code=%d",
                      self.arm_ns, req.gripper_angle, req.gripper_effort, req.gripper_code)
        # reflect opening on joint7/8 (joint7 [0,0.035], joint8 [-0.035,0], symmetric)
        half = max(0.0, min(req.gripper_angle / 2.0, 0.035))
        self.joint_states.position[6] = half
        self.joint_states.position[7] = -half
        response = GripperResponse()
        response.code = 0
        response.status = True
        return response

    ### display loop
    def publish_joints(self, event=None):
        if self.move_t0 is not None:
            alpha = min((rospy.Time.now() - self.move_t0).to_sec() / MOVE_TIME, 1.0)
            self.q_now = self.q_start + alpha * (self.q_target - self.q_start)
            if alpha >= 1.0:
                self.move_t0 = None
                self.q_now = self.q_target.copy()
        self.joint_states.position[0:6] = self.q_now.tolist()
        self.joint_states.header.stamp = rospy.Time.now()
        self.joint_pub.publish(self.joint_states)


if __name__ == '__main__':
    FakePiper()
    rospy.spin()
