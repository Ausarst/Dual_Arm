#!/usr/bin/env python3
""" Fake piper SDK node for dry-run test (no hardware needed)

    Mimic the software interface of piper_ctrl_single_node for one arm:
        Publish   /{arm_ns}/joint_states_single   all-zero joint states at 20Hz
        Service   /{arm_ns}/gripper_srv           log request and return success
        Subscribe /{arm_ns}/pos_cmd               log received cartesian cmd

    Usage:
        rosrun piper fake_piper_node.py _arm_ns:=arm1
        (normally launched by 2arm_dryrun.launch, no need to run by hand)
"""

import rospy
from sensor_msgs.msg import JointState
from piper_msgs.srv import Gripper, GripperResponse
from piper_msgs.msg import PosCmd


class FakePiper:
    def __init__(self):
        rospy.init_node("fake_piper", anonymous=True)
        self.arm_ns = rospy.get_param("~arm_ns", "arm1")

        self.joint_pub = rospy.Publisher(
            "/%s/joint_states_single" % self.arm_ns, JointState, queue_size=1)
        self.joint_states = JointState()
        self.joint_states.name = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
            'joint7', 'joint8']
        self.joint_states.position = [0.0] * 8
        self.joint_states.velocity = [0.0] * 8
        self.joint_states.effort = [0.0] * 8

        rospy.Subscriber("/%s/pos_cmd" % self.arm_ns, PosCmd,
                         self.pos_callback, queue_size=1, tcp_nodelay=True)
        rospy.Service("/%s/gripper_srv" % self.arm_ns, Gripper,
                      self.handle_gripper_service)

        self.pub_timer = rospy.Timer(rospy.Duration(0.05), self.publish_joints)
        rospy.loginfo("fake piper node ready for arm %s", self.arm_ns)

    def publish_joints(self, event=None):
        self.joint_states.header.stamp = rospy.Time.now()
        self.joint_pub.publish(self.joint_states)

    def pos_callback(self, pos_data):
        rospy.loginfo("[%s fake] received PosCmd: x=%.4f y=%.4f z=%.4f r=%.4f p=%.4f yaw=%.4f",
                      self.arm_ns, pos_data.x, pos_data.y, pos_data.z,
                      pos_data.roll, pos_data.pitch, pos_data.yaw)

    def handle_gripper_service(self, req):
        rospy.loginfo("[%s fake] gripper_srv called: angle=%.4f effort=%.2f code=%d",
                      self.arm_ns, req.gripper_angle, req.gripper_effort, req.gripper_code)
        response = GripperResponse()
        response.code = 0
        response.status = True
        return response


if __name__ == '__main__':
    FakePiper()
    rospy.spin()
