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
# from pinocchio import casadi as cpin  # 桥已改为_build_fk_err内探测加载(apt版pinocchio无此桥)

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
        """符号FK误差函数, 两种实现自动选择:
        ① pinocchio-casadi桥(log6误差) —— conda/cmeel版pinocchio才有此桥;
        ② casadi原生DH符号链(位置差+姿态Frobenius差) —— apt版ros-noetic-pinocchio
           未编译casadi桥时的可移植回退; DH参数与SDK piper_fk一致, 已实测与URDF
           link6恒等(T_offset=I), 数值等价。姿态用Frobenius差: 平滑无奇异,
           零当且仅当旋转相等(优化最优解与log6代价一致)。
        两者都预编译成Function供Opti的MX调用。"""
        q_sym = casadi.SX.sym('q', 6)
        T_sym = casadi.SX.sym('T', 4, 4)
        try:
            from pinocchio import casadi as cpin  # noqa: F401 (探测桥是否可用)
            cmodel = cpin.Model(self.kin.model)
            cdata = cmodel.createData()
            cpin.framesForwardKinematics(cmodel, cdata, q_sym)
            oMf = cdata.oMf[self.kin.ee_id]
            err_sym = cpin.log6(oMf.inverse() * cpin.SE3(T_sym)).vector
            self._err_dim, self._ori_slice = 6, slice(3, 6)
            return casadi.Function('fk_err_pin', [q_sym, T_sym], [err_sym])
        except ImportError:
            pass
        # DH符号链(与piper_fk.py dh_is_offset=0x01一致, 单位:米/弧度)
        a = [0, 0, 0.28503, -0.02198, 0, 0]
        alpha = [0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2]
        th0 = [0, np.deg2rad(-172.22), np.deg2rad(-102.78), 0, 0, 0]
        d = [0.123, 0, 0, 0.25075, 0, 0.091]
        T = casadi.SX.eye(4)
        for i in range(6):
            th = q_sym[i] + th0[i]
            ct, st = casadi.cos(th), casadi.sin(th)
            ca_, sa_ = casadi.cos(alpha[i]), casadi.sin(alpha[i])
            # 逐元素复刻SDK piper_fk的__LinkTransformtion(非标准DH布局, d在二三行)
            A = casadi.vertcat(
                casadi.horzcat(ct, -st, 0, a[i]),
                casadi.horzcat(st * ca_, ct * ca_, -sa_, -sa_ * d[i]),
                casadi.horzcat(st * sa_, ct * sa_, ca_, ca_ * d[i]),
                casadi.horzcat(0, 0, 0, 1))
            T = casadi.mtimes(T, A)
        p_err = T[:3, 3] - T_sym[:3, 3]
        R_err = T[:3, :3] - T_sym[:3, :3]      # Frobenius差, 零 iff 旋转相等
        err_sym = casadi.vertcat(p_err, casadi.reshape(R_err, -1, 1))
        self._err_dim, self._ori_slice = 12, slice(3, 12)
        return casadi.Function('fk_err_dh', [q_sym, T_sym], [err_sym])

    def _task_cost(self, err):
        """log6版: 前3位位置/后3位姿态; DH版: 前3位位置/后9位姿态Frobenius"""
        return (self.w_pos * casadi.sumsqr(err[0:3])
                + self.w_ori * casadi.sumsqr(err[self._ori_slice]) / 2.0)

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

        # 终点任务代价: FK(q_N) -> T_goal 误差(位置/姿态加权)
        Tg = casadi.DM(T_goal.homogeneous)
        cost = self._task_cost(self._fk_err(qs[N, :].T, Tg))

        # 可选经由点
        if via_poses:
            for idx, T_via in via_poses:
                cost += self._task_cost(
                    self._fk_err(qs[idx, :].T, casadi.DM(T_via.homogeneous)))

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
