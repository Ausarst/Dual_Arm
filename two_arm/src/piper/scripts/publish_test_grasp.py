#!/usr/bin/env python3
"""
发布一个测试用的 HGGD 抓取位姿到 /hggd_pose_raw
用法：
  rosrun piper publish_test_grasp.py
  rosrun piper publish_test_grasp.py x y z
  rosrun piper publish_test_grasp.py 0.3 0.1 0.2
"""
import rospy
from std_msgs.msg import String
import json
import sys

def publish_grasp(x=0.2, y=0.0, z=0.3, score=0.9, width=0.05):
    # 构造一个单位旋转矩阵的抓取位姿（可根据需要修改）
    grasp_msg = json.dumps([{
        "score": score,
        "width": width,
        "translation": [x, y, z],
        "rotation": [[1,0,0],[0,1,0],[0,0,1]]
    }])
    
    pub = rospy.Publisher('/hggd_pose_raw', String, queue_size=1)
    rospy.init_node('test_grasp_publisher', anonymous=True)
    rospy.sleep(0.5)  # 等待发布者注册
    
    rospy.loginfo("Publishing test grasp: pos=[%.2f, %.2f, %.2f] score=%.2f", x, y, z, score)
    pub.publish(String(data=grasp_msg))
    rospy.sleep(0.5)  # 让消息有时间发送
    rospy.loginfo("Done.")

if __name__ == '__main__':
    # 解析命令行参数
    args = sys.argv[1:]
    x, y, z = 0.2, 0.0, 0.3
    if len(args) == 3:
        try:
            x, y, z = map(float, args)
        except ValueError:
            print("Invalid coordinates, using defaults")
    
    rospy.init_node('test_grasp_publisher', anonymous=True)
    publish_grasp(x, y, z)
