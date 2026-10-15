#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
piper_safe_lift_ramp_test.py

Purpose:
  Slowly test a candidate "lift-away-from-hanging" motion for Piper arm using
  small interpolated /joint_ctrl_single commands.

Why:
  The micro-motion test showed:
    - joint2 positive tracks well.
    - joint3 negative tracks well.
    - joint2 negative is limited/clamped near zero.
    - joint3 positive does not move near current zero.
  Therefore the first lift candidate is usually joint2 positive + joint3 negative.

Safety:
  - Default is dry-run. Add --execute to publish real commands.
  - Does not enable, reset, stop, or go_zero.
  - Reads /joint_states_single and /arm_status.
  - Checks arm_status after each ramp step.
  - Stops if status becomes abnormal.
  - Prompts before execution unless --no-prompt is used.

Example dry-run:
  rosrun piper piper_safe_lift_ramp_test.py

Example very conservative real test:
  rosrun piper piper_safe_lift_ramp_test.py --execute --joint2-total 0.02 --joint3-total -0.02 --step-size 0.002

Example slightly larger after success:
  rosrun piper piper_safe_lift_ramp_test.py --execute --joint2-total 0.05 --joint3-total -0.05 --step-size 0.005
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
        self.items = []
        self.lock = threading.Lock()
        self.max_len = max_len

    def append(self, item):
        with self.lock:
            self.items.append(item)
            if len(self.items) > self.max_len:
                self.items = self.items[-self.max_len:]

    def latest(self):
        with self.lock:
            if not self.items:
                return None
            return self.items[-1]

    def clear(self):
        with self.lock:
            self.items = []

    def snapshot(self):
        with self.lock:
            return list(self.items)


class PiperSafeLiftRampTest(object):
    def __init__(self, args):
        self.args = args
        self.joint_buf = Buffer()
        self.status_buf = Buffer()
        self.results = []
        self.gripper_hold = args.default_gripper

        self.pub = rospy.Publisher(args.cmd_topic, JointState, queue_size=1)
        rospy.Subscriber(args.joint_topic, JointState, self.joint_cb, queue_size=100)

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
        d = dict(zip(msg.name, msg.position))
        if not all(name in d for name in JOINT_NAMES_6):
            return
        joints = [float(d[name]) for name in JOINT_NAMES_6]
        gripper = None
        if "joint7" in d:
            gripper = float(d["joint7"])
        elif "gripper" in d:
            gripper = float(d["gripper"])

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
                if isinstance(v, (bool, int, float, str)):
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
            lim = "joint_%d_angle_limit" % i
            comm = "communication_status_joint_%d" % i
            if hasattr(msg, lim) and bool(getattr(msg, lim)):
                return False
            if hasattr(msg, comm) and bool(getattr(msg, comm)):
                return False
        return True

    def latest_joint(self):
        return self.joint_buf.latest()

    def latest_status(self):
        latest = self.status_buf.latest()
        if latest is None:
            return None, None
        return latest["ok"], latest["summary"]

    def wait_joint(self, timeout=5.0):
        start = time.time()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            item = self.latest_joint()
            if item is not None:
                return item
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for %s" % self.args.joint_topic)
            rate.sleep()

    def mean_current(self, duration=0.5):
        self.joint_buf.clear()
        start = time.time()
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and time.time() - start < duration:
            rate.sleep()
        samples = self.joint_buf.snapshot()
        if not samples:
            item = self.wait_joint()
            return item["joints"]
        sums = [0.0] * 6
        for s in samples:
            for i, v in enumerate(s["joints"]):
                sums[i] += v
        return [v / float(len(samples)) for v in sums]

    def make_msg(self, target):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(CMD_NAMES)
        msg.position = list(target) + [float(self.gripper_hold)]
        msg.velocity = [0, 0, 0, 0, 0, 0, self.args.gripper_velocity]
        msg.effort = [0, 0, 0, 0, 0, 0, self.args.gripper_effort]
        return msg

    def publish_hold(self, target, duration):
        msg = self.make_msg(target)
        rate = rospy.Rate(self.args.pub_rate)
        start = time.time()
        while not rospy.is_shutdown() and time.time() - start < duration:
            msg.header.stamp = rospy.Time.now()
            self.pub.publish(msg)
            rate.sleep()

    def prompt(self, base, final_target, n_steps):
        if not self.args.execute:
            return True
        if self.args.no_prompt:
            return True

        print("\n" + "=" * 80)
        print("准备执行慢速 ramp 测试")
        print("base rad:         %s" % fmt(base))
        print("final target rad: %s" % fmt(final_target))
        print("total delta rad:  %s" % fmt([final_target[i] - base[i] for i in range(6)]))
        print("total delta deg:  %s" % fmt([rad_to_deg(final_target[i] - base[i]) for i in range(6)], 3))
        print("step count:       %d" % n_steps)
        print("请确认：机械臂周围清空、有人守急停/电源、当前已经手动使能。")
        ans = input("执行 ramp？[y/N] ").strip().lower()
        return ans == "y"

    def build_ramp_targets(self, base):
        delta = [0.0] * 6
        delta[0] = self.args.joint1_total
        delta[1] = self.args.joint2_total
        delta[2] = self.args.joint3_total
        delta[3] = self.args.joint4_total
        delta[4] = self.args.joint5_total
        delta[5] = self.args.joint6_total

        max_abs = max(abs(v) for v in delta)
        if max_abs == 0.0:
            raise RuntimeError("All total deltas are zero. Nothing to test.")

        n_steps = int(math.ceil(max_abs / self.args.step_size))
        n_steps = max(1, n_steps)

        targets = []
        for k in range(1, n_steps + 1):
            ratio = float(k) / float(n_steps)
            target = [base[i] + ratio * delta[i] for i in range(6)]
            targets.append(target)
        return targets

    def save_logs(self, base, final_target):
        log_dir = os.path.expanduser(self.args.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        base_path = os.path.join(log_dir, "piper_lift_ramp_%s" % now_stamp())
        csv_path = base_path + ".csv"
        json_path = base_path + ".json"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "target_joint1", "target_joint2", "target_joint3", "target_joint4", "target_joint5", "target_joint6",
                "actual_joint1", "actual_joint2", "actual_joint3", "actual_joint4", "actual_joint5", "actual_joint6",
                "err_joint1", "err_joint2", "err_joint3", "err_joint4", "err_joint5", "err_joint6",
                "status_ok", "status_summary"
            ])
            for r in self.results:
                writer.writerow(
                    [r["step"]] +
                    r["target"] +
                    r["actual"] +
                    r["error"] +
                    [r["status_ok"], json.dumps(r["status_summary"], ensure_ascii=False)]
                )

        data = {
            "args": vars(self.args),
            "base": base,
            "final_target": final_target,
            "results": self.results,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("\n日志已保存:")
        print("  %s" % json_path)
        print("  %s" % csv_path)

    def run(self):
        print("\n等待真实关节反馈: %s" % self.args.joint_topic)
        first = self.wait_joint()
        if first["gripper"] is not None:
            self.gripper_hold = first["gripper"]

        print("首次反馈 rad: %s" % fmt(first["joints"]))
        print("首次反馈 deg: %s" % fmt([rad_to_deg(v) for v in first["joints"]], 3))
        print("gripper hold: %.6f" % self.gripper_hold)

        status_ok, status_summary = self.latest_status()
        print("\n当前 arm_status:")
        print("  status_ok: %s" % str(status_ok))
        print("  summary: %s" % str(status_summary))

        base = self.mean_current(self.args.sample_sec)
        targets = self.build_ramp_targets(base)
        final_target = targets[-1]

        if self.args.execute:
            print("\n当前为 EXECUTE 模式：会发布真实 /joint_ctrl_single 命令。")
        else:
            print("\n当前为 DRY-RUN 模式：不会发布真实命令。")

        if not self.prompt(base, final_target, len(targets)):
            print("用户未确认，退出。")
            return

        if not self.args.execute:
            print("\ndry-run 目标序列:")
            for i, t in enumerate(targets, 1):
                print("  step %02d target rad: %s" % (i, fmt(t)))
            return

        print("\n开始 ramp。")
        for i, target in enumerate(targets, 1):
            print("\nstep %02d/%02d target rad: %s" % (i, len(targets), fmt(target)))
            self.publish_hold(target, self.args.command_sec)
            rospy.sleep(self.args.settle_sec)

            actual = self.mean_current(self.args.sample_sec)
            error = [actual[j] - target[j] for j in range(6)]
            status_ok, status_summary = self.latest_status()

            print("  actual rad: %s" % fmt(actual))
            print("  error rad:  %s" % fmt(error))
            print("  status_ok:  %s" % str(status_ok))

            self.results.append({
                "step": i,
                "target": target,
                "actual": actual,
                "error": error,
                "status_ok": status_ok,
                "status_summary": status_summary,
            })

            if status_ok is False:
                print("\narm_status abnormal. Stop ramp immediately.")
                break

            max_err = max(abs(v) for v in error)
            if max_err > self.args.max_tracking_error:
                print("\ntracking error too large: %.6f rad. Stop ramp." % max_err)
                break

        if self.args.return_base:
            print("\n准备缓慢返回 base。")
            return_targets = list(reversed(targets[:-1])) + [base]
            for i, target in enumerate(return_targets, 1):
                print("return step %02d/%02d target rad: %s" % (i, len(return_targets), fmt(target)))
                self.publish_hold(target, self.args.command_sec)
                rospy.sleep(self.args.settle_sec)
                actual = self.mean_current(self.args.sample_sec)
                status_ok, status_summary = self.latest_status()
                print("  actual rad: %s" % fmt(actual))
                print("  status_ok:  %s" % str(status_ok))
                self.results.append({
                    "step": "return_%02d" % i,
                    "target": target,
                    "actual": actual,
                    "error": [actual[j] - target[j] for j in range(6)],
                    "status_ok": status_ok,
                    "status_summary": status_summary,
                })
                if status_ok is False:
                    print("\narm_status abnormal during return. Stop.")
                    break

        self.save_logs(base, final_target)

        print("\n测试完成。")
        print("建议：如果 joint2_total>0 且 joint3_total<0 的 ramp 平稳，并且肉眼看到末端/相机远离下垂方向，")
        print("就可以把该方向作为后续“自然下垂 -> 小步抬起”的候选启动动作。")


def parse_args():
    p = argparse.ArgumentParser(description="Piper safe lift ramp test")

    p.add_argument("--joint-topic", default="/joint_states_single")
    p.add_argument("--cmd-topic", default="/joint_ctrl_single")
    p.add_argument("--status-topic", default="/arm_status")

    p.add_argument("--execute", action="store_true")
    p.add_argument("--no-prompt", action="store_true")

    p.add_argument("--pub-rate", type=float, default=10.0)
    p.add_argument("--command-sec", type=float, default=0.5)
    p.add_argument("--settle-sec", type=float, default=0.3)
    p.add_argument("--sample-sec", type=float, default=0.5)

    p.add_argument("--joint1-total", type=float, default=0.0)
    p.add_argument("--joint2-total", type=float, default=0.02)
    p.add_argument("--joint3-total", type=float, default=-0.02)
    p.add_argument("--joint4-total", type=float, default=0.0)
    p.add_argument("--joint5-total", type=float, default=0.0)
    p.add_argument("--joint6-total", type=float, default=0.0)

    p.add_argument("--step-size", type=float, default=0.002)
    p.add_argument("--max-tracking-error", type=float, default=0.015)

    p.add_argument("--return-base", action="store_true", help="After ramp, slowly return to base.")
    p.add_argument("--default-gripper", type=float, default=0.00035)
    p.add_argument("--gripper-velocity", type=float, default=10.0)
    p.add_argument("--gripper-effort", type=float, default=0.5)

    p.add_argument("--log-dir", default="~/piper_safety_test_logs")

    return p.parse_args()


def main():
    args = parse_args()
    rospy.init_node("piper_safe_lift_ramp_test")
    node = PiperSafeLiftRampTest(args)
    node.run()


if __name__ == "__main__":
    main()
