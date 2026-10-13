#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

def save_depth_image(msg):
    # 将 ROS Image 转换为 OpenCV 格式
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    # 保存为 PNG
    cv2.imwrite("demo_depth.png", cv_image)
    rospy.loginfo("Saved depth image as depth.png")

def save_color_image(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    cv2.imwrite("demo_rgb.png", cv_image)
    rospy.loginfo("Saved color image as color.png")

def listener():
    rospy.init_node('image_saver', anonymous=True)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, save_depth_image)
    rospy.Subscriber("/camera/color/image_raw", Image, save_color_image)
    rospy.spin()

if __name__ == '__main__':
    listener()

