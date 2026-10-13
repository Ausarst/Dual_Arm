#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Header, Float32MultiArray, String
from piper_msgs.srv import Gripper, GripperRequest
from std_srvs.srv import Empty

# Initialize global variables for axes and buttons
axes_values = [0] * 6  # assuming the joystick has 6 axes
buttons_values = [0] * 12  # assuming the joystick has 12 buttons

# Scaling factors for velocities
# linear_scale = 0.5
# angular_scale = 1.0
linear_scale = 0.02
angular_scale = 0.2
gripper_service = None
hand_service = None


def call_hand_reset():
    em = Empty()
    global hand_service
    hand_service(em)
    print('hand reset')

# Callback function to update joystick axes and button values
def joy_callback(data):
    global axes_values, buttons_values
    axes_values = data.axes  # Update axes values
    buttons_values = data.buttons  # Update button values

def call_gripper_service(gripper_opening, force_effort=2):
    global gripper_service
    try:
        gri_cmd = GripperRequest()
        gri_cmd.gripper_angle = gripper_opening   # 0 - 0.07 m
        gri_cmd.gripper_effort = force_effort     # 0.5 - 2 N
        gri_cmd.set_zero = 0
        gri_cmd.gripper_code = 1
        response = gripper_service(gri_cmd)
        rospy.loginfo("Gripper service called successfully. Response: %s", response)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)
        
def main():
    global gripper_service
    global hand_service
    # Initialize ROS node
    rospy.init_node('twist_publisher', anonymous=True)
    rospy.wait_for_service('/gripper_srv')
    gripper_service = rospy.ServiceProxy('/gripper_srv', Gripper)
    hand_service = rospy.ServiceProxy('/stop_srv', Empty)

    # Create a publisher for the /servo_server/delta_twist_cmds topic
    pub = rospy.Publisher('/servo_server/delta_twist_cmds', TwistStamped, queue_size=10)
    joy_relat_posecmd_pub = rospy.Publisher("joy_relat_posecmd", Float32MultiArray, queue_size=3)
    custom_msg_pub = rospy.Publisher("joy_custom_cmd", String, queue_size=3)
    
    # Subscribe to the joystick's /joy topic
    rospy.Subscriber("/joy", Joy, joy_callback)

    # Set the publishing rate
    rate = rospy.Rate(50)  # 100 Hz

    # Main loop to continuously publish twist messages
    while not rospy.is_shutdown():
        # Create a TwistStamped message
        twist_msg = TwistStamped()
        twist_msg.header = Header()
        twist_msg.header.stamp = rospy.Time.now()

        # Map joystick axes to linear velocities
        twist_msg.twist.linear.x = axes_values[5] * linear_scale  # scale the linear x velocity (e.g., forward/backward)
        twist_msg.twist.linear.y = axes_values[4] * linear_scale  # scale the linear y velocity (e.g., left/right)
        twist_msg.twist.linear.z = 0  # no up/down movement controlled by axes

        # Map joystick buttons to control angular velocities and linear z
        if buttons_values[3] == 1:  # Button 0 increases angular.z (left rotation)
            twist_msg.twist.angular.y = angular_scale
        elif buttons_values[1] == 1:  # Button 1 decreases angular.z (right rotation)
            twist_msg.twist.angular.y = -angular_scale
        else:
            twist_msg.twist.angular.y = 0  # No rotation when neither button is pressed

        if buttons_values[0] == 1:  # Button 2 increases linear.z (move up)
            twist_msg.twist.angular.x = angular_scale
        elif buttons_values[2] == 1:  # Button 3 decreases linear.z (move down)
            twist_msg.twist.angular.x = -angular_scale
        else:
            twist_msg.twist.angular.x = 0  # No vertical movement when neither button is pressed

# Map joystick buttons to control angular.x (pitch)
        if buttons_values[4] == 1:  # Button 4 increases angular.x (pitch forward)
            twist_msg.twist.linear.z = linear_scale
        elif buttons_values[6] == 1:  # Button 5 decreases angular.x (pitch backward)
            twist_msg.twist.linear.z = - linear_scale
        else:
            twist_msg.twist.linear.z = 0  # No angular.x movement when neither button is pressed

        # Map joystick buttons to control angular.y (roll)
        if buttons_values[5] == 1:  # Button 6 increases angular.y (roll right)
            twist_msg.twist.angular.z = angular_scale
        elif buttons_values[7] == 1:  # Button 7 decreases angular.y (roll left)
            twist_msg.twist.angular.z = -angular_scale
        else:
            twist_msg.twist.angular.z = 0  # No angular.y movement when neither button is pressed

# GRIPPER
        if buttons_values[8]:
            call_gripper_service(0)
        if buttons_values[9]:
            call_gripper_service(0.07)

        if buttons_values[10] == 1:
            send = True
            if buttons_values[2] == 1.0:
                rospy.loginfo("R1+A pressed → publish CUST0")
                custom_msg_pub.publish("R1+A")
            elif buttons_values[1] == 1.0:
                rospy.loginfo("R1+B pressed → publish CUST1")
                custom_msg_pub.publish("R1+B")
            elif buttons_values[0] == 1.0:
                rospy.loginfo("R1+Y pressed → publish CUST2")
                custom_msg_pub.publish("R1+Y")
            elif buttons_values[3] == 1.0:
                rospy.loginfo("R1+X pressed → publish CUST3")
                custom_msg_pub.publish("R1+X")
            elif axes_values[3] < -0.95:
                rospy.loginfo("R1+A pressed → publish CUST0")
                custom_msg_pub.publish("R1+A")
            elif axes_values[2] < -0.95:
                rospy.loginfo("R1+B pressed → publish CUST1")
                custom_msg_pub.publish("R1+B")
            elif axes_values[3] > 0.95:
                rospy.loginfo("R1+Y pressed → publish CUST2")
                custom_msg_pub.publish("R1+Y")
            elif axes_values[2] > 0.95:
                rospy.loginfo("R1+Y pressed → publish CUST3")
                custom_msg_pub.publish("R1+X")
            else:
                send = False

            while send and (sum(buttons_values[:4])+sum(axes_values[2:4])**2) != 0: rospy.sleep(0.05)
            continue

        if buttons_values[11] == 1.0:
            rospy.loginfo("GRI_A pressed → publish GRI_A")
            custom_msg_pub.publish("GRI_A")
            # call_hand_reset()

        if twist_msg.twist.linear.x == 0 and twist_msg.twist.linear.y == 0 and twist_msg.twist.linear.z == 0 and \
            twist_msg.twist.angular.x == 0 and twist_msg.twist.angular.y == 0 and twist_msg.twist.angular.z == 0:
            pass
        else:
            ee_rp_cmd_delta_msg = Float32MultiArray()
            ee_rp_cmd_delta_msg.data = [twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z,
                twist_msg.twist.angular.x, twist_msg.twist.angular.y, twist_msg.twist.angular.z]
            joy_relat_posecmd_pub.publish(ee_rp_cmd_delta_msg)
            pub.publish(twist_msg)

            rospy.loginfo(f"Publishing {ee_rp_cmd_delta_msg.data}")
            while (sum(buttons_values) + sum(axes_values)) != 0: rospy.sleep(0.05)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
