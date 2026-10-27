#!/usr/bin/env python3
""" 运动规划A: 直线参考 + 逐点约束QP (抓取接近段专用, 在线流式执行)

    架构(每拍闭环):
        插值(纯数学): SE(3)参考位姿序列 —— 位置直线, 姿态log插值(接近段姿态恒定则退化为纯直线)
        Pinocchio   : 事实层 —— FK当前位姿 / LOCAL雅可比(与log6误差配套) / 自碰撞
        QP(casadi内嵌OSQP): 决策层 ——
            min || J·Δq − e ||² + ε||Δq||²
            s.t. 关节限位 / 关节步长限幅
          解出关节增量后积分推进, 下一拍重算。
        失败兜底: QP不可行或跟踪误差超阈值 -> 立即中止报错(不可行是安全信息, 不松约束硬解)

    输出: 关节轨迹(时间戳对齐20Hz), 执行侧逐点发 /armX/joint_ctrl_single(关节模式,
          固件不再自由插值, 末端严格沿直线)

    用法(tv环境, 与ik_check同):
        python3 motion_a_line_qp.py 0.2506 0.05 0.237 0 1.4838 0    # 从零位形规划到该法兰目标
    参数(节点/CLI外通过LineQPPlanner构造函数调):
        v_lin=0.05 m/s, a_lin=0.2 m/s², rate=20Hz, 关节步长限幅按1.0 rad/s
"""

import sys

import numpy as np
import pinocchio as pin
import casadi

from piper_kin import PiperKin, se3_from_xyzrpy


class LineQPPlanner:
    def __init__(self, kin: PiperKin = None,
                 dt=0.05, v_lin=0.05, a_lin=0.20,
                 qd_max=1.0, eps=1e-4,
                 track_pos_tol=5e-3, track_ori_tol=0.10,
                 inner_iter=8):
        self.kin = kin or PiperKin()
        self.dt = dt
        self.v_lin = v_lin
        self.a_lin = a_lin
        self.qd_max = qd_max          # 关节速度限幅 rad/s
        self.eps = eps                # 阻尼(奇异附近软化)
        self.track_pos_tol = track_pos_tol
        self.track_ori_tol = track_ori_tol
        self.inner_iter = inner_iter   # 每航点QP内层迭代次数

        # 待执行J·Δq逼近当前与期望位姿残差e + 关节角Δq最小变化 = min ‖J·Δq − e‖² + ε‖Δq‖² = min Δqᵀ(JᵀJ + εI)Δq − 2(Jᵀe)ᵀΔq + 常数项(eᵀe) -> mim xᵀPx + 2qvᵀx，I为单位阵，常数项(eᵀe)取决于插值密度
        # s.t. lbx≤Δq≤ubx(关节限位), |Δq|≤lim(限速)
        # 求解器: casadi内嵌OSQP(conic), 不可用则回退qrqp
        x = casadi.SX.sym('dq', 6)
        P = casadi.SX.sym('P', 6, 6)
        qv = casadi.SX.sym('q', 6)
        f = 0.5 * casadi.dot(x, P @ x) + casadi.dot(qv, x)
        # prob导入参数：决策变量x，参数向量p（平铺成一维），目标函数f，约束函数g（对变量x约束即可）
        prob = {'x': x, 'p': casadi.vertcat(casadi.reshape(P, -1, 1), qv),
                'f': f, 'g': x}
        try:
            self._qp = casadi.qpsol('qp', 'osqp', prob,
                                     {'error_on_fail': False, 'verbose': False})
        except Exception:
            self._qp = casadi.qpsol('qp', 'qrqp', prob,
                                     {'error_on_fail': False, 'verbose': False})

    def _solve_step(self, J, e, q):
        P = J.T @ J + self.eps * np.eye(6)
        qlin = -(J.T @ e)
        dq_lim = self.qd_max * self.dt
        sol = self._qp(p=casadi.vertcat(casadi.DM(P).reshape((-1, 1)), casadi.DM(qlin)),
                       lbx=self.kin.q_lo - q, ubx=self.kin.q_hi - q,
                       lbg=-dq_lim, ubg=dq_lim)
        dq_val = np.array(sol['x']).ravel()
        if not np.all(np.isfinite(dq_val)):
            raise RuntimeError("QP infeasible (关节限位/限速与任务冲突)")
        # OSQP容差(1e-3)内可能轻微越界, 钳回可行域
        lo = np.maximum(self.kin.q_lo - q, -dq_lim)
        hi = np.minimum(self.kin.q_hi - q, dq_lim)
        return np.clip(dq_val, lo, hi)

    def _trapezoid_s(self, length):
        """梯形速度profile: 返回每dt的路径参数增量序列(总积分为1)"""
        if length < 1e-9:
            return np.array([1.0])
        t_acc = self.v_lin / self.a_lin
        s_acc = 0.5 * self.a_lin * t_acc**2 / length
        if 2 * s_acc >= 1.0:  # 路径过短, 加速+减速即占满全程: 三角profile
            v_eff = np.sqrt(length * self.a_lin)   # 半程可达峰值 v²=2a·(L/2)
            t_total = 2 * length / v_eff           # 前半加速后半减速, 平均速度v_eff/2
        else:
            v_eff = self.v_lin
            t_total = length / v_eff + t_acc       # ≡ 2t_acc + (L-v·t_acc)/v (含两端ramp的精确总时长)
        # 总时长以dt间隔切分n段时间
        n = max(int(np.ceil(t_total / self.dt)), 2)
        # 插值t个时刻
        t = np.linspace(0.0, t_total, n + 1)
        # 速度曲线: 满加速度爬坡封顶v_eff, 末端自t_total倒排的镜像减速沿, 对称梯形/三角, 与t_total公式逐项一致(加速d_acc+匀速L-2d_acc+减速d_acc=L), 末端零速到达
        # prof就是每个插值t时刻对应的速度
        prof = np.minimum(np.minimum(self.a_lin * t, v_eff),
                          self.a_lin * (t_total - t))
        # ds输出以dt为间隔的位移增量序列，Δt*平均速度v/总长度（归一化）
        ds = np.diff(t) * 0.5 * (prof[:-1] + prof[1:]) / length
        ds = ds / ds.sum()
        return ds

    def plan(self, q_start: np.ndarray, T_goal: pin.SE3,
             T_start: pin.SE3 = None, max_steps=400):
        """返回 (traj: N×6 关节序列, dts: N, stats: dict); 失败抛 RuntimeError"""
        kin = self.kin
        q = np.array(q_start, dtype=float).copy()
        T0 = T_start if T_start is not None else kin.fk(q)
        T1 = T_goal

        # 参考路径: 位置直线 + 姿态log插值
        p0, p1 = T0.translation, T1.translation
        R0, R1 = T0.rotation, T1.rotation
        # w_rot旋转差
        w_rot = pin.log3(R0.T @ R1)
        length = float(np.linalg.norm(p1 - p0))
        ds = self._trapezoid_s(length)

        traj = [q.copy()]
        dts = []
        s = 0.0
        max_dev = 0.0
        # 计算ds每个k时刻需要到达的位姿
        for k in range(len(ds)):
            # s累计自增位移
            s += ds[k]
            # 插值位姿ref按原始位姿乘归一化后的累计自增位移s计算
            T_ref = pin.SE3(R0 @ pin.exp3(s * w_rot), p0 + s * (p1 - p0))

            # 内层迭代: 单次QP受限幅/阻尼约束不一定追平参考, 逐次迭代到收敛(标准做法)
            for _ in range(self.inner_iter):
                T_cur = kin.fk(q)
                e = pin.log6(T_cur.inverse() * T_ref).vector
                if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                    break
                J = kin.jac_local(q)
                try:
                    dq_val = self._solve_step(J, e, q)
                except RuntimeError as ex:
                    raise RuntimeError("step %d: %s" % (k, ex))
                q = q + dq_val
                if kin.self_collision(q):
                    raise RuntimeError("step %d: self collision on planned waypoint" % k)

            # 外层跟踪门限: 内层迭代后仍追不上 -> 参考/限位/奇异冲突, 中止
            e = pin.log6(kin.fk(q).inverse() * T_ref).vector
            if np.linalg.norm(e[:3]) > self.track_pos_tol or \
               np.linalg.norm(e[3:]) > self.track_ori_tol:
                raise RuntimeError(
                    "step %d: tracking lost (pos %.4f ori %.4f), 参考/限位/奇异冲突" %
                    (k, np.linalg.norm(e[:3]), np.linalg.norm(e[3:])))

            traj.append(q.copy())
            dts.append(self.dt)

            # 直线度统计(FK位置到参考直线的偏差)
            pf = kin.fk(q).translation
            if length > 1e-9:
                t_proj = np.clip(np.dot(pf - p0, p1 - p0) / max(length**2, 1e-12), 0, 1)
                max_dev = max(max_dev, float(np.linalg.norm(pf - (p0 + t_proj * (p1 - p0)))))

        # 重定时: 内层迭代使航点间位移可能超过 qd_max·dt(同一时间拍走了多步QP),
        # 按逐关节限速重排各航点时间戳, 保证执行侧按 dts 发送时全局速度不超限
        dq_arr = np.abs(np.diff(np.array(traj), axis=0))
        dts = np.maximum(self.dt, dq_arr.max(axis=1) / self.qd_max).tolist()

        T_end = kin.fk(q)
        pos_err = float(np.linalg.norm(T_end.translation - p1))
        ori_err = float(np.linalg.norm(pin.log3(T_end.rotation.T @ R1)))
        stats = {"steps": len(traj) - 1, "duration": sum(dts),
                 "final_pos_err": pos_err, "final_ori_err": ori_err,
                 "max_line_deviation": max_dev,
                 "max_dq": float(max(np.linalg.norm(np.array(traj[i+1]) - np.array(traj[i]))
                                     for i in range(len(traj) - 1)))}
        return np.array(traj), dts, stats


def main():
    args = [float(a) for a in sys.argv[1:]]
    if len(args) != 6:
        print(__doc__)
        sys.exit(2)
    kin = PiperKin()
    planner = LineQPPlanner(kin)
    T_goal = se3_from_xyzrpy(args[:3], args[3:6])
    traj, dts, stats = planner.plan(np.zeros(6), T_goal)
    print("A(直线+逐点QP) 规划结果:")
    print("  航点数 %d, 时长 %.2fs" % (stats["steps"], stats["duration"]))
    print("  末端直线度: 最大偏离 %.5f m" % stats["max_line_deviation"])
    print("  终点误差: pos %.5f m, ori %.5f rad" % (stats["final_pos_err"], stats["final_ori_err"]))
    print("  最大关节步长 %.4f rad (限幅 %.4f)" % (stats["max_dq"], 1.0 * 0.05))
    print("  末点 q(deg): %s" % np.round(np.degrees(traj[-1]), 1).tolist())


if __name__ == '__main__':
    main()
