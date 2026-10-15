#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
piper_safe_micro_motion_test.py

安全微动测试脚本，用于 Piper 机械臂在竖直安装、自然下垂姿态下的真机调试。

功能：
1. 从 /joint_states_single 自动读取真实当前关节角。
2. 从 /arm_status 自动读取状态码。
3. 通过 /joint_ctrl_single 发布很小的关节测试命令。
4. 每一步自动采样：命令前、命令中、命令后。
5. 自动计算：
   - 目标关节是否真的变化；
   - 实际变化量是多少；
   - 与目标误差是多少；
   - 未命令关节是否发生异常漂移；
   - arm_status / err_code / angle_limit / communication_status 是否异常。
6. 输出 CSV 与 JSON 日志，便于后续写交接报告。

安全设计：
- 默认 dry-run，不会发布真实命令。
- 必须加 --execute 才会真的发布控制命令。
- 默认每一步都要求人工确认。
- 脚本不会调用 enable。
- 脚本不会调用 go_zero。
- 脚本不会调用 reset。
- 脚本不会调用 stop。
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import threading
from datetime import datetime

import rospy
import rostopic
from sensor_msgs.msg import JointState


JOINT_NAMES_6 = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
CMD_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def rad_to_deg(x):
    return x * 180.0 / math.pi


def fmt(vals, nd=6):
    return "[" + ", ".join(("{:." + str(nd) + "f}").format(v) for v in vals) + "]"


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Buffer(object):
    def __init__(self, max_len=10000):
        self.max_len = max_len
        self.items = []
        self.lock = threading.Lock()

    def append(self, item):
        with self.lock:
            self.items.append(item)
            if len(self.items) > self.max_len:
                self.items = self.items[-self.max_len:]

    def clear(self):
        with self.lock:
            self.items = []

    def snapshot(self):
        with self.lock:
            return list(self.items)

    def latest(self):
        with self.lock:
            if not self.items:
                return None
            return self.items[-1]


class PiperSafeMicroMotionTest(object):
    def __init__(self, args):
        self.args = args
        self.joint_buf = Buffer()
        self.status_buf = Buffer()
        self.results = []

        self.gripper_hold = args.default_gripper
        self.initial_base = None
        self.control_base = None

        self.cmd_pub = rospy.Publisher(args.cmd_topic, JointState, queue_size=1)

        rospy.Subscriber(args.joint_topic, JointState, self.joint_cb, queue_size=100)

        # /arm_status 的消息类型可能来自不同包，用 rostopic 动态获取。
        try:
            status_class, real_topic, _ = rostopic.get_topic_class(args.status_topic, blocking=False)
            if status_class is not None:
                rospy.Subscriber(args.status_topic, status_class, self.status_cb, queue_size=100)
                rospy.loginfo("Subscribed to status topic: %s", real_topic)
            else:
                rospy.logwarn("Cannot determine status topic type yet: %s", args.status_topic)
        except Exception as exc:
            rospy.logwarn("Failed to subscribe to %s: %s", args.status_topic, str(exc))

    def joint_cb(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        joints = []
        for name in JOINT_NAMES_6:
            if name not in name_to_pos:
                return
            joints.append(float(name_to_pos[name]))

        gripper = None
        if "joint7" in name_to_pos:
            gripper = float(name_to_pos["joint7"])
        elif "gripper" in name_to_pos:
            gripper = float(name_to_pos["gripper"])

        self.joint_buf.append({
            "t": rospy.Time.now().to_sec(),
            "joints": joints,
            "gripper": gripper,
        })

    def status_cb(self, msg):
        self.status_buf.append({
            "t": rospy.Time.now().to_sec(),
            "ok": self.status_ok(msg),
            "summary": self.status_summary(msg),
        })

    @staticmethod
    def status_summary(msg):
        names = [
            "ctrl_mode", "arm_status", "mode_feedback", "teach_status",
            "motion_status", "trajectory_num", "err_code",
            "joint_1_angle_limit", "joint_2_angle_limit", "joint_3_angle_limit",
            "joint_4_angle_limit", "joint_5_angle_limit", "joint_6_angle_limit",
            "communication_status_joint_1", "communication_status_joint_2",
            "communication_status_joint_3", "communication_status_joint_4",
            "communication_status_joint_5", "communication_status_joint_6",
        ]
        out = {}
        for n in names:
            if hasattr(msg, n):
                v = getattr(msg, n)
                if isinstance(v, (int, float, bool, str)):
                    out[n] = v
                else:
                    out[n] = str(v)
        return out

    @staticmethod
    def status_ok(msg):
        if msg is None:
            return None

        if hasattr(msg, "arm_status") and getattr(msg, "arm_status") != 0:
            return False
        if hasattr(msg, "err_code") and getattr(msg, "err_code") != 0:
            return False

        for i in range(1, 7):
            lim_name = "joint_%d_angle_limit" % i
            comm_name = "communication_status_joint_%d" % i
            if hasattr(msg, lim_name) and bool(getattr(msg, lim_name)):
                return False
            if hasattr(msg, comm_name) and bool(getattr(msg, comm_name)):
                return False

        return True

    def latest_status(self):
        latest = self.status_buf.latest()
        if latest is None:
            return None, None
        return latest["ok"], latest["summary"]

    def wait_first_joint(self, timeout=5.0):
        start = time.time()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            latest = self.joint_buf.latest()
            if latest is not None:
                return latest
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for %s" % self.args.joint_topic)
            rate.sleep()

    def collect_samples(self, duration):
        self.joint_buf.clear()
        start = time.time()
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and time.time() - start < duration:
            rate.sleep()
        samples = self.joint_buf.snapshot()
        if not samples:
            raise RuntimeError("No samples collected from %s" % self.args.joint_topic)
        return samples

    @staticmethod
    def mean_joints(samples):
        s = [0.0] * 6
        for item in samples:
            for i, v in enumerate(item["joints"]):
                s[i] += v
        return [v / float(len(samples)) for v in s]

    @staticmethod
    def std_joints(samples, mean):
        if len(samples) <= 1:
            return [0.0] * 6
        s = [0.0] * 6
        for item in samples:
            for i, v in enumerate(item["joints"]):
                d = v - mean[i]
                s[i] += d * d
        return [math.sqrt(v / float(len(samples) - 1)) for v in s]

    @staticmethod
    def max_abs_delta(samples, ref):
        out = [0.0] * 6
        for item in samples:
            for i, v in enumerate(item["joints"]):
                out[i] = max(out[i], abs(v - ref[i]))
        return out

    def make_cmd(self, target6):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(CMD_NAMES)
        msg.position = list(target6) + [float(self.gripper_hold)]
        msg.velocity = [0, 0, 0, 0, 0, 0, self.args.gripper_velocity]
        msg.effort = [0, 0, 0, 0, 0, 0, self.args.gripper_effort]
        return msg

    def publish_target(self, target6):
        msg = self.make_cmd(target6)
        rate = rospy.Rate(self.args.pub_rate)
        start = time.time()
        while not rospy.is_shutdown() and time.time() - start < self.args.command_sec:
            msg.header.stamp = rospy.Time.now()
            self.cmd_pub.publish(msg)
            rate.sleep()

    def prompt(self, label, target6):
        if not self.args.execute:
            return True
        if self.args.no_prompt:
            return True

        print("\n" + "=" * 80)
        print("准备执行步骤: %s" % label)
        print("目标关节 rad: %s" % fmt(target6))
        print("目标关节 deg: %s" % fmt([rad_to_deg(v) for v in target6], 3))
        print("请确认：机械臂周围清空、有人守急停/电源、当前已经手动使能。")
        ans = input("执行该步骤？[y/N/s跳过/q退出] ").strip().lower()
        if ans == "q":
            print("用户退出。")
            sys.exit(0)
        if ans == "s":
            print("跳过步骤: %s" % label)
            return False
        return ans == "y"

    def analyze(self, label, base, target, before, during, after, commanded_indices):
        before_mean = self.mean_joints(before)
        after_mean = self.mean_joints(after)
        after_std = self.std_joints(after, after_mean)
        during_max = self.max_abs_delta(during, base)

        target_delta = [target[i] - base[i] for i in range(6)]
        achieved_delta = [after_mean[i] - base[i] for i in range(6)]
        target_error = [after_mean[i] - target[i] for i in range(6)]

        cmd_set = set(commanded_indices)
        other_max_drift = 0.0
        other_max_joint = -1
        for i in range(6):
            if i not in cmd_set:
                d = abs(after_mean[i] - base[i])
                if d > other_max_drift:
                    other_max_drift = d
                    other_max_joint = i

        status_ok, status_summary = self.latest_status()

        warnings = []
        failed = False

        if status_ok is False:
            warnings.append("ARM_STATUS_NOT_OK")
            failed = True

        if other_max_drift > self.args.max_other_joint_drift:
            warnings.append("OTHER_JOINT_DRIFT %.6f rad on joint%d" % (other_max_drift, other_max_joint + 1))
            failed = True

        max_std = max(after_std)
        if max_std > self.args.max_settle_std:
            warnings.append("POST_SETTLE_STD %.6f rad" % max_std)

        max_err = max(abs(v) for v in target_error)
        if max_err > self.args.max_tracking_error:
            warnings.append("TRACKING_ERROR %.6f rad" % max_err)

        result = {
            "label": label,
            "base": base,
            "target": target,
            "before_mean": before_mean,
            "after_mean": after_mean,
            "after_std": after_std,
            "during_max_delta_from_base": during_max,
            "target_delta": target_delta,
            "achieved_delta": achieved_delta,
            "target_error": target_error,
            "other_max_drift": other_max_drift,
            "status_ok": status_ok,
            "status_summary": status_summary,
            "warnings": warnings,
            "failed": failed,
        }
        self.results.append(result)
        return result

    def print_result(self, r):
        print("\n结果: %s" % r["label"])
        print("  status_ok: %s" % str(r["status_ok"]))
        print("  base rad:          %s" % fmt(r["base"]))
        print("  target rad:        %s" % fmt(r["target"]))
        print("  after mean rad:    %s" % fmt(r["after_mean"]))
        print("  target delta rad:  %s" % fmt(r["target_delta"]))
        print("  achieved delta:    %s" % fmt(r["achieved_delta"]))
        print("  target error rad:  %s" % fmt(r["target_error"]))
        print("  after std rad:     %s" % fmt(r["after_std"]))
        print("  other max drift:   %.6f rad = %.3f deg" % (
            r["other_max_drift"], rad_to_deg(r["other_max_drift"])))
        if r["warnings"]:
            print("  warnings: %s" % ", ".join(r["warnings"]))
        else:
            print("  warnings: none")

    def run_step(self, label, base, target, commanded_indices):
        if not self.prompt(label, target):
            return None

        print("\n采样命令前状态...")
        before = self.collect_samples(self.args.sample_sec)
        print("  before mean: %s" % fmt(self.mean_joints(before)))

        if not self.args.execute:
            print("  dry-run：没有发布真实命令。")
            result = self.analyze(label, base, target, before, before, before, commanded_indices)
            self.print_result(result)
            return result

        print("发布命令 %.2f 秒..." % self.args.command_sec)
        self.joint_buf.clear()
        self.publish_target(target)
        during = self.joint_buf.snapshot()
        if not during:
            during = before

        print("等待稳定 %.2f 秒..." % self.args.settle_sec)
        rospy.sleep(self.args.settle_sec)

        print("采样命令后状态...")
        after = self.collect_samples(self.args.sample_sec)

        result = self.analyze(label, base, target, before, during, after, commanded_indices)
        self.print_result(result)

        if result["failed"] and self.args.stop_on_fail:
            print("\n保守安全检查失败，停止后续测试。")
            self.save_logs()
            sys.exit(2)

        return result

    def save_logs(self):
        log_dir = os.path.expanduser(self.args.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        base = os.path.join(log_dir, "piper_micro_motion_%s" % now_stamp())
        json_path = base + ".json"
        csv_path = base + ".csv"

        data = {
            "args": vars(self.args),
            "initial_base": self.initial_base,
            "control_base": self.control_base,
            "gripper_hold": self.gripper_hold,
            "results": self.results,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "joint", "base", "target", "after_mean",
                "target_delta", "achieved_delta", "target_error",
                "after_std", "status_ok", "warnings"
            ])
            for r in self.results:
                for i, name in enumerate(JOINT_NAMES_6):
                    writer.writerow([
                        r["label"], name,
                        r["base"][i], r["target"][i], r["after_mean"][i],
                        r["target_delta"][i], r["achieved_delta"][i],
                        r["target_error"][i], r["after_std"][i],
                        r["status_ok"], "|".join(r["warnings"])
                    ])

        print("\n日志已保存:")
        print("  %s" % json_path)
        print("  %s" % csv_path)

    def run(self):
        print("\n等待真实关节反馈: %s" % self.args.joint_topic)
        first = self.wait_first_joint()
        if first["gripper"] is not None:
            self.gripper_hold = first["gripper"]

        print("首次反馈:")
        print("  joints rad: %s" % fmt(first["joints"]))
        print("  joints deg: %s" % fmt([rad_to_deg(v) for v in first["joints"]], 3))
        print("  gripper: %.6f" % self.gripper_hold)

        status_ok, status_summary = self.latest_status()
        print("\n当前 arm_status:")
        print("  status_ok: %s" % str(status_ok))
        print("  summary: %s" % str(status_summary))

        if self.args.execute:
            print("\n当前为 EXECUTE 模式：会发布真实 /joint_ctrl_single 命令。")
        else:
            print("\n当前为 DRY-RUN 模式：不会发布真实命令。")

        print("\n采样 initial_base...")
        init_samples = self.collect_samples(self.args.sample_sec)
        self.initial_base = self.mean_joints(init_samples)
        print("initial_base rad: %s" % fmt(self.initial_base))
        print("initial_base deg: %s" % fmt([rad_to_deg(v) for v in self.initial_base], 3))

        # 第一步：当前位置保持，用来检测首次控制接管跳动。
        self.run_step(
            "00_hold_initial_takeover_check",
            self.initial_base,
            self.initial_base,
            []
        )

        print("\n采样 control_base，即首次接管后的基准姿态...")
        base_samples = self.collect_samples(self.args.sample_sec)
        self.control_base = self.mean_joints(base_samples)
        print("control_base rad: %s" % fmt(self.control_base))
        print("control_base deg: %s" % fmt([rad_to_deg(v) for v in self.control_base], 3))

        d_wrist = self.args.delta_wrist
        d_lift = self.args.delta_lift

        steps = [
            ("01_hold_control_base", {}, []),

            ("02_joint5_plus_%.4f" % d_wrist, {4: +d_wrist}, [4]),
            ("03_return_after_joint5_plus", {}, []),
            ("04_joint5_minus_%.4f" % d_wrist, {4: -d_wrist}, [4]),
            ("05_return_after_joint5_minus", {}, []),

            ("06_joint2_plus_%.4f" % d_lift, {1: +d_lift}, [1]),
            ("07_return_after_joint2_plus", {}, []),
            ("08_joint2_minus_%.4f" % d_lift, {1: -d_lift}, [1]),
            ("09_return_after_joint2_minus", {}, []),

            ("10_joint3_plus_%.4f" % d_lift, {2: +d_lift}, [2]),
            ("11_return_after_joint3_plus", {}, []),
            ("12_joint3_minus_%.4f" % d_lift, {2: -d_lift}, [2]),
            ("13_return_after_joint3_minus", {}, []),
        ]

        if self.args.include_combo:
            steps.extend([
                ("14_combo_joint2_plus_joint3_plus", {1: +d_lift, 2: +d_lift}, [1, 2]),
                ("15_return_after_combo", {}, []),
            ])

        for label, offsets, cmd_indices in steps:
            target = list(self.control_base)
            for idx, offset in offsets.items():
                target[idx] += offset
            self.run_step(label, self.control_base, target, cmd_indices)

        self.save_logs()

        print("\n测试完成。")
        print("解释建议:")
        print("1. 如果 00_hold_initial 有跳动，但 01_hold_control_base 不跳，说明首次控制接管存在一次性对齐。")
        print("2. 如果 joint2/joint3 正负方向都 status_ok=True，可以用 achieved_delta 判断控制是否准确。")
        print("3. 肉眼判断方向：哪个方向让末端/相机远离下垂方向，哪个方向才是后续抬起候选。")
        print("4. 在日志稳定前，不要直接扩大到大幅观察姿态。")


def parse_args():
    p = argparse.ArgumentParser(description="Piper safe micro-motion test")

    p.add_argument("--joint-topic", default="/joint_states_single")
    p.add_argument("--cmd-topic", default="/joint_ctrl_single")
    p.add_argument("--status-topic", default="/arm_status")

    p.add_argument("--execute", action="store_true", help="真正发布控制命令")
    p.add_argument("--no-prompt", action="store_true", help="每一步不再询问确认，不建议首次使用")
    p.add_argument("--stop-on-fail", action="store_true", default=True)

    p.add_argument("--pub-rate", type=float, default=10.0)
    p.add_argument("--command-sec", type=float, default=2.0)
    p.add_argument("--settle-sec", type=float, default=1.0)
    p.add_argument("--sample-sec", type=float, default=1.0)

    p.add_argument("--delta-wrist", type=float, default=0.005)
    p.add_argument("--delta-lift", type=float, default=0.010)
    p.add_argument("--include-combo", action="store_true")

    p.add_argument("--max-other-joint-drift", type=float, default=0.010)
    p.add_argument("--max-settle-std", type=float, default=0.004)
    p.add_argument("--max-tracking-error", type=float, default=0.012)

    p.add_argument("--default-gripper", type=float, default=0.00035)
    p.add_argument("--gripper-velocity", type=float, default=10.0)
    p.add_argument("--gripper-effort", type=float, default=0.5)

    p.add_argument("--log-dir", default="~/piper_safety_test_logs")

    return p.parse_args()


def main():
    args = parse_args()
    rospy.init_node("piper_safe_micro_motion_test")
    node = PiperSafeMicroMotionTest(args)
    node.run()


if __name__ == "__main__":
    main()
