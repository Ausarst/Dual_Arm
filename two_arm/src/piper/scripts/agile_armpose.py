#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Float32MultiArray
import copy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from piper_msgs.msg import PosCmd
from omni_msgs.msg import OmniButtonEvent
from tf import transformations
import time
from piper_msgs.srv import Gripper, GripperRequest # newcode

class ArmposeServer:
    def __init__(self, hz):
        rospy.init_node("armpose_server")
        rospy.loginfo("armpose_server running.")
        self.hz = hz
        self.sub_joint_states = rospy.Subscriber('joint_states_single', JointState, self.joint_state_cb, queue_size=1)
        self.sub_custom_cmd = rospy.Subscriber("joy_custom_cmd", String, self.handle_joy_custom_cmd, queue_size=10)
        self.sub_pose = rospy.Subscriber("/phantom/phantom/pose", PoseStamped, self.touch_pose_cb, queue_size=1)
        self.sub_button = rospy.Subscriber("/phantom/phantom/button", OmniButtonEvent, self.touch_button_cb, queue_size=1)
        self.sub_arm_pose = rospy.Subscriber("/end_pose", PoseStamped, self.arm_pose_cb, queue_size=1)
        self.arm_pose_pub = rospy.Publisher("/pos_cmd", PosCmd, queue_size=1)
        self.arm_joint_pub = rospy.Publisher("/joint_ctrl_single", JointState, queue_size=1)
        self.pub_pose_vis_act = rospy.Publisher("/vis_act_pose", PoseStamped, queue_size=1)
        self.pub_pose_vis_touchct = rospy.Publisher("/vis_touch_ct_pose", PoseStamped, queue_size=1)
        self.joy_relat_posecmd = rospy.Subscriber("joy_relat_posecmd", Float32MultiArray, self.joy_relat_posecmd_cb, queue_size=3)
        self.gripper_service = rospy.ServiceProxy('/gripper_srv', Gripper) # newcode

        self.arm_pose = None
        self.touch_pose = None
        self.update_arm = False
        self.update_touch = False
        self.arm_pose_ref = None
        self.touch_pose_ref = None
        self.touch_button_grey = False
        self.touch_button_white = False
        self.act_mode = False
        self.custom_pose_ongoing = False
        self._gripper = 0
        self.joint_current = None
        self.wait_delt_pose_time = time.time()


    def joint_state_cb(self, msg):
        self.joint_current = msg.position
        # print('gripper_current', gripper_current)
        
    def handle_joy_custom_cmd(self, msg: String):
        joint_list = [[0, 0.20, 0.08, 0.0, 0.35, 0.0],
            [0, 0.86, -0.505, -0.0, 0.45, 0.0],
            [0, 1.8, -1.0, 0.0, 0.8, 0.0],
            [0, 2.2, -1.2, 0.0, 0.65, 0.0]]
        #    0 -2.9 -1.9 0 0 0 0
        
        rospy.loginfo("[armpose_server]: %s", msg.data)

        if msg.data == "R1+X":
            self.arm_servo_joint_nogripper(joint_list[0])
        elif msg.data == "R1+Y":
            self.arm_servo_joint_nogripper(joint_list[1])
        elif msg.data == "R1+B":
            self.arm_servo_joint_nogripper(joint_list[2])
        elif msg.data == "R1+A":
            self.arm_servo_joint_nogripper(joint_list[3])
        elif msg.data == "GRI_A": # newcode
            self.gripper_servo(0.019, 2) # newcode

    def gripper_servo(self, gripper_opening, force_effort=2): # newcode
        gri_cmd = GripperRequest()
        gri_cmd.gripper_angle = gripper_opening # 0 - 0.07 m
        gri_cmd.gripper_effort = force_effort # 0.5 - 2 N
        gri_cmd.set_zero = 0
        gri_cmd.gripper_code = 1
        self.gripper_service(gri_cmd)
        self._gripper = gripper_opening
# rostopic pub /gripper_srv your_pkg/GripperRequest \
# "{gripper_angle: 0.05, gripper_effort: 1.0, set_zero: 0, gripper_code: 1}"

    def joy_relat_posecmd_cb(self, msg):
        if self.arm_pose is None:
            rospy.logwarn("arm_pose is None, cannot process joy_relat_posecmd")
            return
        print('delta_pose', msg.data)
        arm_pose_curr = self.arm_pose
        joy_euler = msg.data[3:6]
        arm_pose_euler = transformations.euler_from_quaternion([arm_pose_curr.pose.orientation.x,
                                                                    arm_pose_curr.pose.orientation.y,
                                                                    arm_pose_curr.pose.orientation.z,
                                                                    arm_pose_curr.pose.orientation.w])
        arm_pose_new_euler = [arm_pose_euler[0] + joy_euler[0],
                                   arm_pose_euler[1] + joy_euler[1],
                                   arm_pose_euler[2] + joy_euler[2]]
        joy_pos = msg.data[0:3]
        arm_pose_new_pos = [arm_pose_curr.pose.position.x + joy_pos[0],
                            arm_pose_curr.pose.position.y + joy_pos[1],
                            arm_pose_curr.pose.position.z + joy_pos[2]]
        self.arm_servo_pose(arm_pose_new_pos + arm_pose_new_euler)

    def rotate_orientation_x_90_ori(self, pose_msg):
        # Extract position
        pos = [pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z]

        # Extract orientation quaternion
        quat = [pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w]

        # Convert quaternion to transformation matrix
        T = transformations.quaternion_matrix(quat)
        T[0:3, 3] = pos  # set translation

        # Build rotation matrix for 90° around x-axis
        Rx = transformations.rotation_matrix(1.57, (1, 0, 0))
        Rx2 = transformations.rotation_matrix(-1.57, (0, 0, 1))
        Rx3 = transformations.rotation_matrix(3.1415926, (0, 1, 0))
        Rx4 = transformations.rotation_matrix(3.1415926, (0, 0, 1))
        new_Ta = transformations.concatenate_matrices(Rx2, Rx, T)
        new_Tb = transformations.concatenate_matrices(new_Ta, Rx3, Rx4)

        new_posb = transformations.translation_from_matrix(new_Tb)
        new_quatb = transformations.quaternion_from_matrix(new_Tb)
        new_pose = PoseStamped()
        new_pose.header = pose_msg.header
        new_pose.pose.position.x = new_posb[2]
        new_pose.pose.position.y = new_posb[1]
        new_pose.pose.position.z = -new_posb[0]
        new_pose.pose.orientation.x = new_quatb[0]
        new_pose.pose.orientation.y = new_quatb[1]
        new_pose.pose.orientation.z = new_quatb[2]
        new_pose.pose.orientation.w = new_quatb[3]
        return new_pose

    def touch_pose_cb(self, msg):
        self.touch_pose_prev = copy.deepcopy(msg)
        # self.touch_pose = self.rotate_pose_x_90_matrix(self.touch_pose_prev)
        self.touch_pose = self.rotate_orientation_x_90_ori(self.touch_pose_prev)
        self.pub_pose_vis_touchct.publish(self.touch_pose)
        self.update_touch = True

    def arm_pose_cb(self, msg):
        self.arm_pose = copy.deepcopy(msg)
        self.update_arm = True

    def touch_button_cb(self, msg):
        self.touch_button_grey = (msg.grey_button == 1)
        self.touch_button_white = (msg.white_button == 1)

    def arm_servo_pose(self, act_pose):
        pos_data = PosCmd()
        pos_data.x = act_pose[0]
        pos_data.y = act_pose[1]
        pos_data.z = act_pose[2]
        pos_data.roll = act_pose[3]
        pos_data.pitch = act_pose[4]
        pos_data.yaw = act_pose[5]
        pos_data.gripper = -1 # no use
        pos_data.mode1 = 1  # 位置模式
        pos_data.mode2 = 0  # 速度模式
        self.arm_pose_pub.publish(pos_data)

    def arm_servo_joint(self, act_joint):
        joint_states = JointState()
        joint_states.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
        joint_states.position = act_joint
        joint_states.velocity = [0,0,0,0,0,0,10]
        joint_states.effort = [0,0,0,0,0,0,0.5]
        self.arm_joint_pub.publish(joint_states)

    def arm_servo_joint_nogripper(self, act_joint):
        joint_states = JointState()
        joint_states.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_states.position = act_joint
        joint_states.velocity = [0,0,0,0,0,0]
        joint_states.effort = [0,0,0,0,0,0]
        self.arm_joint_pub.publish(joint_states)

    def reset_pose_reference(self):
        if self.update_touch or self.update_arm:
            self.arm_pose_ref = copy.deepcopy(self.arm_pose)
            self.touch_pose_ref = copy.deepcopy(self.touch_pose)
            self.update_touch, self.update_arm = False, False
            self.act_x, self.act_y, self.act_z = 0, 0, 0
            # rospy.loginfo('reset_pose_reference.: {}'.format(self.ur_pose_ref))
            # rospy.loginfo('reset_pose_reference.: {}'.format(self.touch_pose_ref))

    def touch_to_arm_mapping(self):
        if (self.arm_pose_ref is None) or (self.touch_pose_ref is None) or (self.touch_pose is None): 
            rospy.logwarn("Some of below pose is missing:")
            rospy.logwarn("arm_pose_ref:{}, touch_pose_ref:{}, touch_pose:{}".format(
                self.arm_pose_ref, self.touch_pose_ref, self.touch_pose))
            return
        touch_ref_curr = copy.deepcopy(self.touch_pose_ref)
        touch_new_curr = copy.deepcopy(self.touch_pose)
        # scaling_factor = 1.5 * 1000 # m to mm and scaling
        # scaling_factor = 1
        scaling_factor = 2
        self.act_x = (touch_new_curr.pose.position.x - touch_ref_curr.pose.position.x)
        self.act_y = (touch_new_curr.pose.position.y - touch_ref_curr.pose.position.y)
        self.act_z = (touch_new_curr.pose.position.z - touch_ref_curr.pose.position.z)
        threshold_move = 0.1*0.1
        if (self.act_x**2 + self.act_y**2 + self.act_z**2) > threshold_move:
            rospy.logwarn(f"too large move : {self.act_x}, {self.act_y}, {self.act_z} m, skipping...")
            return

        ## feasibility of orientation
        qa = (touch_new_curr.pose.orientation.x, 
                touch_new_curr.pose.orientation.y, 
                touch_new_curr.pose.orientation.z, 
                touch_new_curr.pose.orientation.w)
        # qut = transformations.quaternion_multiply([-0.7071068, 0, 0, 0.7071068], qa)
        qut = qa
        act_roll, act_pitch, act_yaw = transformations.euler_from_quaternion(qut)
        # act_yaw = - act_yaw
        # act_roll, act_pitch = 0, 0

        act_pose = [self.arm_pose_ref.pose.position.x + (self.act_x * scaling_factor),
                    self.arm_pose_ref.pose.position.y + (self.act_y * scaling_factor),
                    self.arm_pose_ref.pose.position.z + (self.act_z * scaling_factor),
                    act_roll, act_pitch, act_yaw]
        self.arm_servo_pose(act_pose)
        self.visualize_act_pose(act_pose)

        rospy.loginfo('touch_to_arm_mapping: {}'.format(act_pose))
        # diff_q = transformations.quaternion_from_euler(roll, pitch, yaw)

    def touch_arm_servo(self):
        loop_rate = rospy.Rate(self.hz)
        while not rospy.is_shutdown():
            if self.touch_button_grey or self.touch_button_white:
                if self.touch_button_grey:
                    self.act_mode = not self.act_mode
                if self.touch_button_white:
                    if self._gripper <= 0.032: self._gripper = 0.07
                    else: self._gripper = 0
                    # self._gripper = not self._gripper
                    self.gripper_servo(self._gripper)

                while(self.touch_button_grey or self.touch_button_white): 
                    rospy.sleep(rospy.Duration(0.2))
                if self.act_mode:
                    # self.move_pose_match_touch_ori()
                    self.reset_pose_reference()

            elif self.touch_button_white:
                if self.act_mode:
                    pass
            elif self.touch_button_grey:
                if self.act_mode:
                    pass
            else:
                pass

            if self.act_mode:
                self.touch_to_arm_mapping()
            loop_rate.sleep()

    def visualize_act_pose(self, act_pose):
        visual = PoseStamped()
        visual.header.frame_id = "map"
        visual.pose.position.x = act_pose[0]
        visual.pose.position.y = act_pose[1]
        visual.pose.position.z = act_pose[2]
        qut = transformations.quaternion_from_euler(act_pose[3], act_pose[4], act_pose[5])
        # qut = transformations.quaternion_multiply([-0.7071068, 0, 0, 0.7071068],qa)
        visual.pose.orientation.x = qut[0]
        visual.pose.orientation.y = qut[1]
        visual.pose.orientation.z = qut[2]
        visual.pose.orientation.w = qut[3]
        self.pub_pose_vis_act.publish(visual)


if __name__ == "__main__":
    hz = 30 
    # if len(sys.argv)==1 else int(sys.argv[1])
    rospy.loginfo("servo for robotarm {} hz".format(hz))
    app = ArmposeServer(hz)
    app.touch_arm_servo()

