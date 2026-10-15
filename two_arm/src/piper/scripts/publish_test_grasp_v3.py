#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publish_test_grasp_v3.py

发布测试用 HGGD 抓取候选到 /hggd_pose_raw。

用法：
  # 默认在 camera_color_optical_frame 下发点
  rosrun piper publish_test_grasp_v3.py 0.2 0.0 0.3 --multi

  # 直接在 arm_base 下发点，绕过相机 TF，专门测试 MoveIt 可达性
  rosrun piper publish_test_grasp_v3.py 0.45 0.0 0.25 --frame arm_base --multi

说明：
- HGGD 真实输出通常是 camera_color_optical_frame。
- --frame arm_base 只用于仿真调试，可以快速判断某个 arm_base 目标点是否可达。
"""

import argparse
import json
import rospy
from std_msgs.msg import String


def make_grasp(x, y, z, frame_id, score=0.9, width=0.05, rot=None):
    if rot is None:
        rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return {
        "score": float(score),
        "width": float(width),
        "translation": [float(x), float(y), float(z)],
        "rotation": rot,
        "frame_id": frame_id
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float, nargs="?", default=0.2)
    parser.add_argument("y", type=float, nargs="?", default=0.0)
    parser.add_argument("z", type=float, nargs="?", default=0.3)
    parser.add_argument("--frame", type=str, default="camera_color_optical_frame",
                        help="camera_color_optical_frame or arm_base")
    parser.add_argument("--score", type=float, default=0.9)
    parser.add_argument("--width", type=float, default=0.05)
    parser.add_argument("--multi", action="store_true")
    args = parser.parse_args()

    rospy.init_node("test_grasp_publisher_v3", anonymous=True)
    pub = rospy.Publisher("/hggd_pose_raw", String, queue_size=1)
    rospy.sleep(0.5)

    if args.multi:
        grasps = [
            make_grasp(args.x, args.y, args.z, args.frame, score=args.score, width=args.width),
            make_grasp(args.x - 0.05, args.y, args.z + 0.03, args.frame, score=args.score - 0.05, width=args.width),
            make_grasp(args.x, args.y + 0.05, args.z, args.frame, score=args.score - 0.10, width=args.width),
            make_grasp(args.x, args.y - 0.05, args.z, args.frame, score=args.score - 0.15, width=args.width),
        ]
    else:
        grasps = [make_grasp(args.x, args.y, args.z, args.frame, score=args.score, width=args.width)]

    msg = json.dumps(grasps)
    rospy.loginfo("Publishing %d grasp candidate(s) in frame=%s: %s", len(grasps), args.frame, msg)
    pub.publish(String(data=msg))
    rospy.sleep(0.5)
    rospy.loginfo("Done.")


if __name__ == "__main__":
    main()
