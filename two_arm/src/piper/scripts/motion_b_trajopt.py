#!/usr/bin/env python3
""" 运动规划B: 轨迹优化器 (独立可复用, 路径在约束下自然生成, 适用非直线运动)

    与A的区别: A预先给定直线参考路径, B只锚定起点与终点任务目标,
    中间关节航点全部是优化变量 —— 路径形状由 "任务代价 + 平滑代价 + 约束" 自然涌现,
    Pinocchio(casadi符号) 把 FK 写进目标函数, ipopt 求解整条轨迹。

    形式化 (决策变量 q_1..q_N ∈ R^{N×6}, q_0 = q_start 固定):
        min   Σ ||log6(FK(q_N)⁻¹·T_goal)||²_W      (末端任务, 默认仅终点锚定)
            + w_smooth·Σ||q_{k+1} − 2q_k + q_{k−1}||²   (平滑/最小抖动)
        s.t.  q_lo ≤ q_k ≤ q_hi                    (关节限位)
              |q_{k+1} − q_k| ≤ qd_max·dt          (速度限幅)
    可选 via_poses: 在指定航点索引附加位姿经过约束(非直线成形/绕行)。

    用法(tv环境):
        python3 motion_b_trajopt.py 0.2506 0.05 0.237 0 1.4838 0
    复用: from motion_b_trajopt import TrajOptimizer; traj, stats = TrajOptimizer().solve(q0, T_goal)
"""

import sys

import numpy as np
import pinocchio as pin
import casadi
from pinocchio import casadi as cpin

from piper_kin import PiperKin, se3_from_xyzrpy


class TrajOptimizer:
    def __init__(self, kin: PiperKin = None, N=40, dt=0.05,
                 qd_max=1.0, w_pos=10.0, w_ori=1.0, w_smooth=1.0,
                 max_iter=300):
        self.kin = kin or PiperKin()
        self.N = N
        self.dt = dt
        self.qd_max = qd_max
        self.w_pos = w_pos
        self.w_ori = w_ori
        self.w_smooth = w_smooth
        self.max_iter = max_iter

    def _build_fk_err(self):
        """SX上预编译 FK误差函数(pinocchio-casadi桥只支持SX, Opti变量是MX,
        预编译Function后可用MX调用 —— 与Arm_IK/ik_check同款做法)"""
        cmodel = cpin.Model(self.kin.model)
        cdata = cmodel.createData()
        q_sym = casadi.SX.sym('q', 6)
        T_sym = casadi.SX.sym('T', 4, 4)
        cpin.framesForwardKinematics(cmodel, cdata, q_sym)
        oMf = cdata.oMf[self.kin.ee_id]
        err_sym = cpin.log6(oMf.inverse() * cpin.SE3(T_sym)).vector
        return casadi.Function('fk_err', [q_sym, T_sym], [err_sym])

    def solve(self, q_start: np.ndarray, T_goal: pin.SE3,
              via_poses=None, verbose=False):
        """via_poses: 可选 [(index, pin.SE3), ...] 在指定航点附加位姿经过约束。
        返回 (traj N+1×6, stats); 失败抛 RuntimeError"""
        N, dt = self.N, self.dt
        nq = 6

        if not hasattr(self, '_fk_err'):
            self._fk_err = self._build_fk_err()

        opti = casadi.Opti()
        qs = opti.variable(N + 1, nq)
        opti.subject_to(qs[0, :].T == np.array(q_start).ravel())

        # 关节限位 + 速度限幅
        opti.subject_to(opti.bounded(self.kin.q_lo, qs.T, self.kin.q_hi))
        dq_step = self.qd_max * dt
        opti.subject_to(opti.bounded(-dq_step, qs[1:, :].T - qs[:-1, :].T, dq_step))

        # 终点任务代价: FK(q_N) -> T_goal 的log6误差(位置/姿态加权)
        Tg = casadi.DM(T_goal.homogeneous)
        err = self._fk_err(qs[N, :].T, Tg)
        cost = self.w_pos * casadi.sumsqr(err[0:3]) + self.w_ori * casadi.sumsqr(err[3:6])

        # 可选经由点
        if via_poses:
            for idx, T_via in via_poses:
                ev = self._fk_err(qs[idx, :].T, casadi.DM(T_via.homogeneous))
                cost += self.w_pos * casadi.sumsqr(ev[0:3]) + self.w_ori * casadi.sumsqr(ev[3:6])

        # 平滑代价(二阶差分)
        acc = qs[2:N + 1, :] - 2 * qs[1:N, :] + qs[0:N - 1, :]
        cost = cost + self.w_smooth * casadi.sumsqr(acc)

        opti.minimize(cost)
        opti.set_initial(qs, np.tile(np.array(q_start, dtype=float).ravel(), (N + 1, 1)))
        opti.solver('ipopt', {'ipopt': {'print_level': 0 if not verbose else 5,
                                        'max_iter': self.max_iter},
                              'print_time': False})
        try:
            sol = opti.solve()
        except RuntimeError:
            raise RuntimeError("轨迹优化不收敛: 尝试增大N/降qd_max/检查目标可达性")
        traj = np.array(sol.value(qs))

        # 后验校验: 终点误差 / 限位 / 速度 / 逐步自碰撞
        T_end = self.kin.fk(traj[-1])
        pos_err = float(np.linalg.norm(T_end.translation - T_goal.translation))
        ori_err = float(np.linalg.norm(pin.log3(T_end.rotation.T @ T_goal.rotation)))
        if pos_err > 3e-3 or ori_err > 0.05:
            raise RuntimeError("优化收敛但终点误差过大 (pos %.4f ori %.4f)" % (pos_err, ori_err))
        dq_all = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        for k, q in enumerate(traj):
            if self.kin.self_collision(q):
                raise RuntimeError("航点 %d 自碰撞" % k)
        stats = {"steps": N, "duration": N * dt,
                 "final_pos_err": pos_err, "final_ori_err": ori_err,
                 "max_step": float(dq_all.max()),
                 "smoothness": float(np.abs(np.diff(traj, 2, axis=0)).max())}
        return traj, stats


def main():
    args = [float(a) for a in sys.argv[1:]]
    if len(args) != 6:
        print(__doc__)
        sys.exit(2)
    import time
    kin = PiperKin()
    opt = TrajOptimizer(kin)
    T_goal = se3_from_xyzrpy(args[:3], args[3:6])
    t0 = time.time()
    traj, stats = opt.solve(np.zeros(6), T_goal)
    print("B(轨迹优化) 结果: 求解耗时 %.2fs" % (time.time() - t0))
    print("  航点数 %d, 时长 %.2fs" % (stats["steps"], stats["duration"]))
    print("  终点误差: pos %.5f m, ori %.5f rad" % (stats["final_pos_err"], stats["final_ori_err"]))
    print("  最大关节步长 %.4f rad, 二阶差分最大 %.4f" % (stats["max_step"], stats["smoothness"]))
    print("  末点 q(deg): %s" % np.round(np.degrees(traj[-1]), 1).tolist())


if __name__ == '__main__':
    main()
