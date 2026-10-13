#!/usr/bin/env python3
""" Standalone IK reachability & self-collision checker for piper PosCmd (NO ROS, NO hardware)

    The piper SDK cartesian convention (PosCmd / end_pose, DH based) differs from
    the URDF link6 frame by a constant transform T_offset. This script calibrates
    T_offset numerically against the SDK's own FK (piper_sdk kinematics piper_fk.py)
    and checks targets given in the SDK convention:
        - IK solvable within joint limits?  (pos err < 5mm, ori err < 0.05 rad)
        - self collision at the solution?
    Solver math extracted from piper_pinocchio/piper_pinocchio.py Arm_IK
    (same casadi/ipopt weights, locked joints, ee frame), minus meshcat/real arm.

    Usage (conda env tv, see piper_pinocchio/README.MD):
        conda activate tv
        python3 ik_check.py 0.30 0.00 0.25 0 0 0   0.40 0.00 0.25 0 0 0
        (any number of x y z roll pitch yaw targets, m & rad)
        exit code 0 = all targets reachable & collision free
"""

import os
import sys

import numpy as np
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as SR

CUR = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(os.path.dirname(os.path.dirname(CUR)),
                    'piper_description/urdf/piper_description.urdf')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(CUR)),
                                'piper_sdk', 'piper_sdk', 'kinematics'))
from piper_fk import C_PiperForwardKinematics  # noqa: E402

POS_ERR_TOL = 0.005   # m
ORI_ERR_TOL = 0.05    # rad


def se3_from_xyzrpy(xyz, rpy):
    return pin.SE3(SR.from_euler('xyz', rpy).as_matrix(), np.array(xyz))


class IKCheck:
    def __init__(self, urdf_path=URDF):
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        # lock virtual gripper joints, same as Arm_IK
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=["joint7", "joint8"],
            reference_configuration=np.array([0] * self.robot.model.nq),
        )
        # end effector frame on joint6, identity rotation, same as Arm_IK
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(pin.Quaternion(1, 0, 0, 0),
                              np.array([0.0, 0.0, 0.0])),
                      pin.FrameType.OP_FRAME)
        )
        # wrapper data was created before addFrame, rebuild so oMf covers it
        self.reduced_robot.data = self.reduced_robot.model.createData()
        self.ee_id = self.reduced_robot.model.getFrameId("ee")

        # collision pairs, same as Arm_IK
        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        for i in range(4, 10):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        # casadi symbolic IK, same weights/limits/opts as Arm_IK
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.error = casadi.Function(
            "error", [self.cq, self.cTf],
            [casadi.vertcat(
                cpin.log6(self.cdata.oMf[self.ee_id].inverse()
                          * cpin.SE3(self.cTf)).vector)])
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        error_vec = self.error(self.var_q, self.param_tf)
        self.totalcost = (casadi.sumsqr(1.0 * error_vec[:3])
                          + casadi.sumsqr(0.1 * error_vec[3:]))
        self.regularization = casadi.sumsqr(self.var_q)
        # regularization pulls solutions toward zero pose (same as Arm_IK),
        # keep it for coarse multistart but disable in the polish pass,
        # otherwise a persistent ~5mm / ~0.05rad residual equilibrium appears
        self.param_reg = self.opti.parameter(1)
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit))
        self.opti.minimize(20 * self.totalcost + self.param_reg * self.regularization)
        self.opti.solver("ipopt", {'ipopt': {'print_level': 0, 'max_iter': 100, 'tol': 1e-6},
                                   'print_time': False})

        # calibrate constant transform between SDK cartesian frame and URDF ee frame
        self.sdk_fk = C_PiperForwardKinematics()
        T_urdf0 = self.fk_urdf_ee(np.zeros(6))
        T_sdk0 = self.fk_sdk(np.zeros(6))
        self.T_offset = T_urdf0.inverse() * T_sdk0   # SDK = URDF_ee * T_offset
        self.calib_err = self.check_offset_constancy()

    def fk_urdf_ee(self, q):
        pin.framesForwardKinematics(self.reduced_robot.model,
                                    self.reduced_robot.data, q)
        return self.reduced_robot.data.oMf[self.ee_id].copy()

    def fk_sdk(self, q):
        x, y, z, r, p, yw = self.sdk_fk.CalFK(list(q))[5]
        return se3_from_xyzrpy([x / 1000.0, y / 1000.0, z / 1000.0],
                               np.deg2rad([r, p, yw]))

    def check_offset_constancy(self):
        """T_offset must be constant if URDF and SDK DH describe the same chain"""
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(5):
            q = rng.uniform(-1.0, 1.0, 6)
            T = self.fk_urdf_ee(q).inverse() * self.fk_sdk(q)
            worst = max(worst, np.abs(T.homogeneous - self.T_offset.homogeneous).max())
        return worst

    def check_self_collision(self, q, gripper=np.array([0.0, 0.0])):
        pin.forwardKinematics(self.robot.model, self.robot.data,
                              np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data,
                                     self.geom_model, self.geometry_data)
        return pin.computeCollisions(self.geom_model, self.geometry_data, False)

    def check(self, x, y, z, roll, pitch, yaw):
        """IK for one target in the SDK cartesian frame.
        Multi-start against singular local minima (cold start from zeros often
        gets stuck, the original Arm_IK relied on warm-start history).
        return (ok, sol_q, pos_err, ori_err, collision)"""
        T_sdk_target = se3_from_xyzrpy([x, y, z], [roll, pitch, yaw])
        # solver works in the URDF ee frame
        T_urdf_target = T_sdk_target * self.T_offset.inverse()

        lower = self.reduced_robot.model.lowerPositionLimit
        upper = self.reduced_robot.model.upperPositionLimit
        rng = np.random.default_rng(1)
        guesses = [np.zeros(self.reduced_robot.model.nq)] + \
            [rng.uniform(lower, upper) for _ in range(7)]

        best = None
        for guess in guesses:
            self.opti.set_initial(self.var_q, guess)
            self.opti.set_value(self.param_tf, T_urdf_target.homogeneous)
            self.opti.set_value(self.param_reg, 0.01)
            try:
                self.opti.solve_limited()
                q = self.opti.value(self.var_q)
            except Exception:
                continue
            T_got = self.fk_urdf_ee(q) * self.T_offset
            pos_err = np.linalg.norm(T_got.translation - T_sdk_target.translation)
            ori_err = np.linalg.norm(pin.log3(T_got.rotation @ T_sdk_target.rotation.T))
            score = pos_err + 0.1 * ori_err
            if best is None or score < best[0]:
                best = (score, q, pos_err, ori_err)
        if best is None:
            return False, None, None, None, None
        # polish: warm restart from best, regularization off, for final precision
        _, sol_q, pos_err, ori_err = best
        self.opti.set_initial(self.var_q, sol_q)
        self.opti.set_value(self.param_tf, T_urdf_target.homogeneous)
        self.opti.set_value(self.param_reg, 0.0)
        try:
            self.opti.solve_limited()
            q2 = self.opti.value(self.var_q)
            T_got = self.fk_urdf_ee(q2) * self.T_offset
            pos2 = np.linalg.norm(T_got.translation - T_sdk_target.translation)
            ori2 = np.linalg.norm(pin.log3(T_got.rotation @ T_sdk_target.rotation.T))
            if pos2 + 0.1 * ori2 < pos_err + 0.1 * ori_err:
                sol_q, pos_err, ori_err = q2, pos2, ori2
        except Exception:
            pass

        collision = self.check_self_collision(sol_q)
        ok = (pos_err < POS_ERR_TOL) and (ori_err < ORI_ERR_TOL) and (not collision)
        return ok, sol_q, pos_err, ori_err, collision


def main():
    args = [float(a) for a in sys.argv[1:]]
    if len(args) % 6 != 0 or not args:
        print(__doc__)
        sys.exit(2)
    targets = [args[i:i + 6] for i in range(0, len(args), 6)]

    checker = IKCheck()
    print("frame calibration: max T_offset variation over random q = %.2e (should be ~0)"
          % checker.calib_err)
    if checker.calib_err > 1e-3:
        print("WARNING: SDK FK and URDF chains disagree, results unreliable!")

    all_ok = True
    for idx, t in enumerate(targets):
        ok, q, pos_err, ori_err, collision = checker.check(*t)
        print("target %d: x=%.3f y=%.3f z=%.3f r=%.3f p=%.3f yaw=%.3f" % (idx, *t))
        if q is None:
            print("  FAIL: IK solver did not converge")
            all_ok = False
            continue
        qdeg = np.degrees(q).round(2).tolist()
        print("  sol q(deg): %s" % qdeg)
        print("  pos_err=%.4f m  ori_err=%.4f rad  self_collision=%s"
              % (pos_err, ori_err, collision))
        print("  %s" % ("PASS: reachable & collision free" if ok else "FAIL"))
        all_ok = all_ok and ok
    code = 0 if all_ok else 1
    sys.stdout.flush()
    os._exit(code)  # skip interpreter teardown, pinocchio cleanup can abort


if __name__ == '__main__':
    main()
