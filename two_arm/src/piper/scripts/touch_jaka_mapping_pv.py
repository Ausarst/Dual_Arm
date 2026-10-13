#! /usr/bin/env python

import jaka_msgs.srv
import rospy
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TwistStamped
from omni_msgs.msg import OmniFeedback
from omni_msgs.msg import OmniButtonEvent
import tf
import copy
import jaka_msgs
import math

class touch_jaka_pose_node:
    def __init__(self, hz):
        # self.pose_pub = rospy.Publisher("/pose_servo_cmd", PoseStamped, queue_size=1)
        # self.sub_pose = rospy.Subscriber("/ur_pose_state", PoseStamped, self.ur_pose_cb, queue_size=1)
        self.sub_pose = rospy.Subscriber("/phantom/phantom/pose", PoseStamped, self.touch_pose_cb, queue_size=1)
        self.pub_motor_speed = rospy.Publisher("/SCS_speed", Int32, queue_size=1)
        self.pub_force_feedback = rospy.Publisher("/phantom/phantom/force_feedback", OmniFeedback, queue_size=1)
        self.pub_pose_vis_jaka = rospy.Publisher("/vis_jaka_pose", PoseStamped, queue_size=1)
        self.pub_pose_vis_rtouch = rospy.Publisher("/vis_rtouch_pose", PoseStamped, queue_size=1)
        self.sub_button = rospy.Subscriber("/phantom/phantom/button", OmniButtonEvent, self.touch_button_cb, queue_size=1)
        self.joint_position = rospy.Subscriber("/jaka_driver/joint_position", JointState, self.jaka_joints_cb, queue_size=1)
        self.tool_position = rospy.Subscriber("/jaka_driver/tool_position", TwistStamped, self.jaka_pose_cb, queue_size=10)
        rospy.wait_for_service('/jaka_driver/servo_move_enable')
        self.servo_enable = rospy.ServiceProxy('/jaka_driver/servo_move_enable', jaka_msgs.srv.ServoMoveEnable)
        self.servo_j = rospy.ServiceProxy('/jaka_driver/servo_j', jaka_msgs.srv.ServoMove)
        self.servo_p = rospy.ServiceProxy('/jaka_driver/servo_p', jaka_msgs.srv.ServoMove)
        self.linear_move = rospy.ServiceProxy('/jaka_driver/linear_move', jaka_msgs.srv.Move)
        self.joint_move = rospy.ServiceProxy('/jaka_driver/joint_move', jaka_msgs.srv.Move)
        self.get_jaka_tcp = rospy.ServiceProxy('/jaka_driver/get_tcp', jaka_msgs.srv.GetTCP)
        self.hz = hz
        self.motor_speed = None
        self.jaka_pose = None
        self.jaka_joints = None
        self.touch_pose = None
        self.update_jaka = False
        self.update_touch = False
        self.jaka_pose_ref = None
        self.touch_pose_ref = None
        self.touch_button_grey = False
        self.touch_button_white = False
        self.jaka_servo_flag = False
        self.act_mode = False

    def motor_speed_set(self, speed):
        if self.motor_speed == speed: return
        self.pub_motor_speed.publish(Int32(speed))
        self.motor_speed = speed

    def touch_force_feedback(self, force, position):
        # rostopic pub /phantom/phantom/force_feedback omni_msgs/OmniFeedback
        # "{force: {x: 1.0, y: .0, z: 0.0}, position: {x: 0.240, y: 0.150, z: 0.160}}"
        fp = OmniFeedback()
        fp.force.x = force[0]
        fp.force.y = force[1]
        fp.force.z = force[2]
        fp.position.x = position[0]
        fp.position.y = position[1]
        fp.position.z = position[2]
        self.pub_force_feedback.publish(fp)

    def jaka_enable_servo(self, flag):
        if flag != self.jaka_servo_flag:
            if flag == False: 
                self.jaka_servo_flag = flag
                rospy.sleep(rospy.Duration(0.5))
            self.servo_enable.call(jaka_msgs.srv.ServoMoveEnableRequest(flag))
            self.jaka_servo_flag = flag
            rospy.sleep(rospy.Duration(0.1))

    def jaka_servo_j(self, sj):
        joints = jaka_msgs.srv.ServoMoveRequest()
        joints.pose = sj
        joints.speed = [0.50] * 6
        self.servo_j(joints)

    def jaka_servo_p(self, sp):
        spose = jaka_msgs.srv.ServoMoveRequest()
        spose.pose = sp
        spose.speed = [10] * 6
        self.servo_p(spose)
        
    def jaka_joint_move(self, js):
        jm = jaka_msgs.srv.MoveRequest()
        jm.pose = js
        jm.has_ref = False
        jm.ref_joint = [0]
        jm.mvvelo = 1.5
        jm.mvacc = 1.5
        jm.mvtime = 0.0
        jm.mvradii = 0.0
        jm.coord_mode = 0
        jm.index = 0
        self.joint_move(jm)

    def jaka_linear_move(self, lm):
        jm = jaka_msgs.srv.MoveRequest()
        jm.pose = lm
        jm.has_ref = False
        jm.ref_joint = [0]
        jm.mvvelo = 100
        jm.mvacc = 100
        jm.mvtime = 0.0
        jm.mvradii = 0.0
        jm.coord_mode = 0
        jm.index = 0
        self.linear_move(jm)

    def touch_button_cb(self, msg):
        self.touch_button_grey = (msg.grey_button == 1)
        self.touch_button_white = (msg.white_button == 1)

    def jaka_joints_cb(self, msg):
        self.jaka_joints = copy.deepcopy(list(msg.position))
        self.update_jaka_joints = True
     
    def touch_pose_cb(self, msg):
        self.touch_pose = copy.deepcopy(msg)
        self.update_touch = True

    def jaka_pose_cb(self, msg):
        self.jaka_pose = copy.deepcopy(msg)
        self.update_jaka = True

    def reset_pose_reference(self):
        if self.update_touch or self.update_jaka:
            self.jaka_pose_ref = copy.deepcopy(self.jaka_pose)
            self.touch_pose_ref = copy.deepcopy(self.touch_pose)
            self.update_touch, self.update_jaka = False, False
            self.act_x, self.act_y, self.act_z = 0, 0, 0
            # rospy.loginfo('reset_pose_reference.: {}'.format(self.ur_pose_ref))
            # rospy.loginfo('reset_pose_reference.: {}'.format(self.touch_pose_ref))

    def touch_to_jaka_mapping(self):
        # self.jaka_pose = self.get_jaka_tcp(True)
        if (self.jaka_pose_ref is None) or (self.touch_pose_ref is None) or (self.touch_pose is None): 
            rospy.logwarn("Some of below pose is missing:")
            rospy.logwarn("ur_pose_ref:{}, touch_pose_ref:{}, touch_pose:{}".format(
                self.jaka_pose_ref, self.touch_pose_ref, self.touch_pose))
            return
        touch_ref_curr = copy.deepcopy(self.touch_pose_ref)
        touch_new_curr = copy.deepcopy(self.touch_pose)
        force_scale = -20
        scaling_factor = 1.5 * 1000 # m to mm and scaling
        self.act_x = -(touch_new_curr.pose.position.x - touch_ref_curr.pose.position.x)
        self.act_y = -(touch_new_curr.pose.position.y - touch_ref_curr.pose.position.y)
        # force_scale = -50
        # scaling_factor = 20 # m to mm and scaling
        # self.act_x -= touch_new_curr.pose.position.x - touch_ref_curr.pose.position.x
        # self.act_y -= touch_new_curr.pose.position.y - touch_ref_curr.pose.position.y
        self.act_z += touch_new_curr.pose.position.z - touch_ref_curr.pose.position.z
        self.act_z = 0

        self.touch_force_feedback(force=[force_scale * (touch_new_curr.pose.position.x - touch_ref_curr.pose.position.x),
                                         force_scale * (touch_new_curr.pose.position.y - touch_ref_curr.pose.position.y),
                                         force_scale * (touch_new_curr.pose.position.z - touch_ref_curr.pose.position.z),
                                         ], 
                                position=[touch_ref_curr.pose.position.x, 
                                          touch_ref_curr.pose.position.y, 
                                          touch_ref_curr.pose.position.z])

        ## feasibility of orientation
        qa = (touch_new_curr.pose.orientation.x, 
                touch_new_curr.pose.orientation.y, 
                touch_new_curr.pose.orientation.z, 
                touch_new_curr.pose.orientation.w)
        qut = tf.transformations.quaternion_multiply([-0.7071068, 0, 0, 0.7071068], qa)
        act_roll, act_pitch, act_yaw = tf.transformations.euler_from_quaternion(qut)
        act_yaw = - act_yaw
        act_roll, act_pitch = 0, 0

        act_pose = [self.jaka_pose_ref.twist.linear.x + (self.act_x * scaling_factor),
                    self.jaka_pose_ref.twist.linear.y + (self.act_y * scaling_factor),
                    self.jaka_pose_ref.twist.linear.z + (self.act_z * scaling_factor),
                    act_roll, act_pitch, act_yaw]
        self.jaka_servo_p(act_pose)
        rospy.loginfo('touch_to_ur_mapping: {}'.format(act_pose))
        # diff_q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

    def test_joint_move_servo(self):
        loop_rate = rospy.Rate(self.hz)
        self.jaka_enable_servo(True)
        while(not self.update_jaka_joints): loop_rate.sleep()
        js = self.jaka_joints
        incre_e = 0.002
        incre = incre_e
        while not rospy.is_shutdown():
            # self.jaka_joint_move(js)
            self.jaka_servo_j(js)
            rospy.loginfo("sent jionts {}".format(js))
            js[5] += incre
            if (js[5] > 1.57): incre = - incre_e
            if (js[5] < -1.47): incre = incre_e
        self.jaka_enable_servo(False)

    def test_linear_servo(self):
        loop_rate = rospy.Rate(self.hz)
        self.jaka_enable_servo(True)
        while(not self.update_jaka): loop_rate.sleep()
        jk_pose = [self.jaka_pose.twist.linear.x, 
                   self.jaka_pose.twist.linear.y, 
                   self.jaka_pose.twist.linear.z,
                   self.jaka_pose.twist.angular.x / 180 * math.pi, 
                   self.jaka_pose.twist.angular.y / 180 * math.pi, 
                   self.jaka_pose.twist.angular.z / 180 * math.pi]
        incre_e = 0.05
        incre = incre_e
        while not rospy.is_shutdown():
            self.jaka_servo_p(jk_pose)
            rospy.loginfo("sent jk_pose {}".format(jk_pose))
            jk_pose[2] += incre
            if (jk_pose[2] > 450): incre = - incre_e
            if (jk_pose[2] < 350): incre = incre_e
        self.jaka_enable_servo(False)

    def move_robot_match_touch_ori(self):
        posec = self.get_jaka_tcp(True)
        print('posec', posec)
        qa = (self.touch_pose.pose.orientation.x, 
                self.touch_pose.pose.orientation.y, 
                self.touch_pose.pose.orientation.z, 
                self.touch_pose.pose.orientation.w)
        qut = tf.transformations.quaternion_multiply([-0.7071068, 0, 0, 0.7071068], qa)
        act_roll, act_pitch, act_yaw = tf.transformations.euler_from_quaternion(qut)
        act_yaw = - act_yaw
        act_roll, act_pitch = 0, 0
        pose_target = [posec.pose[0], posec.pose[1], posec.pose[2], act_roll, act_pitch, act_yaw]
        print('pose_target', pose_target)
        self.jaka_linear_move(pose_target)
        rospy.loginfo("Moved to target match, start servo.")
    
    def touch_jaka_servo(self):
        loop_rate = rospy.Rate(self.hz)
        while not rospy.is_shutdown():
            self.visualize_jaka_touch_pose()

            if self.touch_button_grey and self.touch_button_white: 
                self.act_mode = not self.act_mode
                self.motor_speed_set(0)
                while(self.touch_button_grey and self.touch_button_white): 
                    rospy.sleep(rospy.Duration(0.1))
                if self.act_mode:
                    self.jaka_enable_servo(False)
                    self.move_robot_match_touch_ori()
                    self.reset_pose_reference()
                self.jaka_enable_servo(self.act_mode)

            elif self.touch_button_white:
                if self.act_mode or True:
                    self.motor_speed_set(300)
            elif self.touch_button_grey:
                if self.act_mode or True:
                    self.motor_speed_set(-300)
            else:
                self.motor_speed_set(0)

            if self.act_mode:
                self.touch_to_jaka_mapping()
            loop_rate.sleep()

    def visualize_jaka_touch_pose(self):
        # self.jaka_pose = self.get_jaka_tcp(True)
        # print(self.jaka_pose.pose)
        # visual = PoseStamped()
        # visual.header = rospy.Header(frame_id = 'map')
        # visual.pose.position.x = self.jaka_pose.pose[0] / 1000
        # visual.pose.position.y = self.jaka_pose.pose[1] / 1000
        # visual.pose.position.z = self.jaka_pose.pose[2] / 1000
        # qut = tf.transformations.quaternion_from_euler(self.jaka_pose.pose[3], self.jaka_pose.pose[4], self.jaka_pose.pose[5])
        # visual.pose.orientation.x = qut[0]
        # visual.pose.orientation.y = qut[1]
        # visual.pose.orientation.z = qut[2]
        # visual.pose.orientation.w = qut[3]
        # self.pub_pose_vis_jaka.publish(visual)
        if self.jaka_pose is None: return
        visual = PoseStamped()
        visual.header = rospy.Header(frame_id = 'map')
        visual.pose.position.x = self.jaka_pose.twist.linear.x / 1000
        visual.pose.position.y = self.jaka_pose.twist.linear.y / 1000
        visual.pose.position.z = self.jaka_pose.twist.linear.z / 1000
        qut = tf.transformations.quaternion_from_euler(self.jaka_pose.twist.angular.x / 180 * math.pi, 
                                                 self.jaka_pose.twist.angular.y / 180 * math.pi,
                                                 self.jaka_pose.twist.angular.z / 180 * math.pi)
        visual.pose.orientation.x = qut[0]
        visual.pose.orientation.y = qut[1]
        visual.pose.orientation.z = qut[2]
        visual.pose.orientation.w = qut[3]
        self.pub_pose_vis_jaka.publish(visual)

        visual = copy.deepcopy(self.touch_pose)
        qa = (visual.pose.orientation.x, visual.pose.orientation.y, visual.pose.orientation.z, visual.pose.orientation.w)
        qut = tf.transformations.quaternion_multiply([-0.7071068, 0, 0, 0.7071068],qa)
        visual.pose.orientation.x = qut[0]
        visual.pose.orientation.y = qut[1]
        visual.pose.orientation.z = qut[2]
        visual.pose.orientation.w = qut[3]
        self.pub_pose_vis_rtouch.publish(visual)


if __name__ == "__main__":
    import sys
    rospy.init_node('pose_to_servo_touch')
    rospy.loginfo("Initialized node")
    hz = 30 if len(sys.argv)==1 else int(sys.argv[1])
    rospy.loginfo("servo for robotarm {} hz".format(hz))
    app = touch_jaka_pose_node(hz)
    app.touch_jaka_servo()
    # app.test_linear_servo()
    # app.test_joint_move_servo()