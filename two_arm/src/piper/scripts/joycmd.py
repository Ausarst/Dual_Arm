#!/usr/bin/env python3
import sys
import rospy
import time
from functools import partial
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from typing import Any
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

class RobotReceiver:
  def __init__(self):
    self.active_flag = True
    self.cmd_vel_pub = rospy.Publisher("joy_cmd_vel", Twist, queue_size=3)
    self.joy_relat_posecmd_pub = rospy.Publisher("joy_relat_posecmd", Float32MultiArray, queue_size=3)
    self.custom_msg_pub = rospy.Publisher("joy_custom_cmd", String, queue_size=3)
  
    self.joystick_btn_map = {
      "A": 0,
      "B": 1,
      "X": 2,
      "Y": 3,
      "L1": 4,
      "R2": 5,
      "L2": 6,
      "R1": 7,
      "SELECT": 8,
      "START": 9,
      "U": 12,
      "D": 13,
      "L": 14,
      "R": 15,
      "MENU": 16,
      "BACK": 17
    }
    self._warned_no_map = False

  # 获取按钮值
  def _btn(self, sensor_joy: datatypes.SensorJoy, name: str) -> float:
    buttons: Any = getattr(sensor_joy, 'buttons', None)
    if isinstance(buttons, dict):
      return float(buttons.get(name, 0.0))
    if isinstance(buttons, (list, tuple)):
      idx = self.joystick_btn_map.get(name)
      if isinstance(idx, int) and 0 <= idx < len(buttons):
        try:
          return float(buttons[idx])
        except Exception:
          return 0.0
    if not self._warned_no_map and not self.joystick_btn_map:
      rospy.logwarn_throttle(5.0, "sensorjoy: no /sensorjoy/button_map provided; all buttons treated as 0")
      self._warned_no_map = True
    return 0.0

  # 用于接收遥控器数据的回调函数
  def sensorJoyCallback(self, sensor_joy: datatypes.SensorJoy):
    if self.active_flag == True:
      self.active_flag = False
    else: 
      return True
    # Twist 发布（axes: [left_horizon=0, left_vertical=1, right_horizon=2, right_vertical=3]）
    axes = getattr(sensor_joy, 'axes', [])
    def ax(i: int) -> float:
      try:
        return float(axes[i])
      except Exception:
        return 0.0
    twist_cmd = Twist()
    twist_cmd.linear.x = ax(1) * 0.5
    twist_cmd.linear.y = ax(0) * 0.5
    twist_cmd.angular.z = ax(2) * 0.5
    self.cmd_vel_pub.publish(twist_cmd)

    # 构造按钮值
    U = self._btn(sensor_joy, "U"); D = self._btn(sensor_joy, "D"); L = self._btn(sensor_joy, "L"); R = self._btn(sensor_joy, "R")
    L2 = self._btn(sensor_joy, "L2"); R2 = self._btn(sensor_joy, "R2"); R1 = self._btn(sensor_joy, "R1")
    X = self._btn(sensor_joy, "X"); B = self._btn(sensor_joy, "B"); Y = self._btn(sensor_joy, "Y"); A = self._btn(sensor_joy, "A")

    L1 = self._btn(sensor_joy, "L1")
    # 初始化 7 维增量: dx, dy, dz, dR, dP, dY, gripper
    # dx = dy = dz = dR = dP = dY = 0.0

    # # 平移映射
    # # if R2 == 0.0:
    # pos_step = 0.005
    # dy = pos_step * L - pos_step * R
    # if L2 == 1.0:
    #   dx = 0.0
    #   dz = pos_step * U - pos_step * D
    # elif L1 == 1.0 :
    #   dx = pos_step* U - pos_step * D
    #   dz = 0.0

    # # 旋转映射
    # if R1 == 0:
    #   dP = pos_step * Y - pos_step* A
    # if L2 == 1.0:
    #   dR = 0.0
    #   dY = pos_step * X - pos_step * B
    # else:
    #   dR = pos_step * X - pos_step * B
    #   dY = 0.0

    # # 夹爪（开=1/关=0）
    # gripper = R2

    # 自定义消息发布逻辑
    if R1 == 1.0:
        if A == 1.0:
            rospy.loginfo("R1+A pressed → publish TOCAMPOSE")
            self.custom_msg_pub.publish("R1+A")
        elif B == 1.0:
            rospy.loginfo("R1+B pressed → publish INITPOSE")
            self.custom_msg_pub.publish("R1+B")
        elif Y == 1.0:
            rospy.loginfo("R1+Y pressed → publish CUST2")
            self.custom_msg_pub.publish("R1+Y")
        elif X == 1.0:
            rospy.loginfo("R1+Y pressed → publish CUST2")
            self.custom_msg_pub.publish("R1+X")
        # else:
        #     rospy.loginfo("R1 pressed → publish CUST3")
        #     self.custom_msg_pub.publish("CUST3")

        # rospy.sleep(0.1)

    # if dx==0 and dy==0 and dz==0 and dR==0 and dP==0 and dY==0:
    #    return
    # ee_rp_cmd_delta_msg = Float32MultiArray()
    # ee_rp_cmd_delta_msg.data = [dx, dy, dz, dR, dP, dY, gripper]
    # self.joy_relat_posecmd_pub.publish(ee_rp_cmd_delta_msg)
    # return True

if __name__ == '__main__':
    # 创建一个 Robot 实例，类型为 PointFoot
    robot = Robot(RobotType.PointFoot)
    rospy.init_node("sensorjoy_node")
    print('running node')

    robot_ip = "10.192.1.2"
    # 检查是否提供了机器人 IP 的命令行参数
    # if len(sys.argv) > 1:
    #     robot_ip = sys.argv[1]

    # 使用 robot_ip 初始化机器人
    if not robot.init(robot_ip):
        sys.exit()

    # 创建一个 RobotReceiver 实例来处理回调
    receiver = RobotReceiver()

    # 创建回调函数的 partial 函数
    sensorJoyCallback = partial(receiver.sensorJoyCallback)

    # 订阅机器人遥控数据
    robot.subscribeSensorJoy(sensorJoyCallback)
    
    # 休眠 1 秒，防止程序退出
    import time
    hz = 50
    while True:
      receiver.active_flag = True
      time.sleep(1/hz)
    
