#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publish_test_grasp_v2.py

发布测试用 HGGD 抓取候选到 /hggd_pose_raw。

用法：
  rosrun piper publish_test_grasp_v2.py
  rosrun piper publish_test_grasp_v2.py x y z
  rosrun piper publish_test_grasp_v2.py x y z --multi

说明：
- x y z 是 camera_color_optical_frame 下的坐标。
- --multi 会同时发布多个候选，用来测试 v2 节点的“逐个候选规划筛选”逻辑。
"""

import argparse
import json
import rospy
from std_msgs.msg import String


def make_grasp(x, y, z, score=0.9, width=0.05, rot=None):
    if rot is None:
        rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return {
        "score": float(score),
        "width": float(width),
        "translation": [float(x), float(y), float(z)],
        "rotation": rot
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float, nargs="?", default=0.2)
    parser.add_argument("y", type=float, nargs="?", default=0.0)
    parser.add_argument("z", type=float, nargs="?", default=0.3)
    parser.add_argument("--score", type=float, default=0.9)
    parser.add_argument("--width", type=float, default=0.05)
    parser.add_argument("--multi", action="store_true")
    args = parser.parse_args()

    rospy.init_node("test_grasp_publisher", anonymous=True)
    pub = rospy.Publisher("/hggd_pose_raw", String, queue_size=1)
    rospy.sleep(0.5)

    if args.multi:
        grasps = [
            make_grasp(args.x, args.y, args.z, score=args.score, width=args.width),
            make_grasp(args.x + 0.04, args.y, args.z, score=args.score - 0.05, width=args.width),
            make_grasp(args.x, args.y + 0.04, args.z, score=args.score - 0.10, width=args.width),
        ]
    else:
        grasps = [make_grasp(args.x, args.y, args.z, score=args.score, width=args.width)]

    msg = json.dumps(grasps)
    rospy.loginfo("Publishing %d test grasp candidate(s): %s", len(grasps), msg)
    pub.publish(String(data=msg))
    rospy.sleep(0.5)
    rospy.loginfo("Done.")


if __name__ == "__main__":
    main()
