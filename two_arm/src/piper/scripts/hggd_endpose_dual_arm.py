""" /grasp_pose
    def pub_grasp_pose(self, pred_gg):
        self.pred_gg = pred_gg
        grasps = [] 
        for g in self.pred_gg: 
            grasps.append({ 
                "score": float(g.score), 
                "width": float(g.width), 
                "height": float(g.height), 
                "depth": float(g.depth), 
                "translation": g.translation.tolist(), 
                "rotation": g.rotation.tolist() }) 
        self.pub.publish(json.dumps(grasps)) 
"""

""" todo:
1. Subscribe raw pose from HGGD
2. Transform pose to PoseStamped and translate them from camera base to arm base by TF
3. Publish
"""

import rospy
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs
import tf2_geometry_msgs.tf2_geometry_msgs
import geometry_msgs
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf import transformations
import json
# from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState

from typing import Union
import numpy as np
from scipy.spatial.transform import Rotation as SR

from piper_msgs.srv import Gripper, GripperRequest
from piper_msgs.msg import PosCmd
gripper_service = None

class HGGDCallback:
    def __init__(self):
        rospy.init_node("hggd_pose_process")
        rospy.Subscriber("/hggd_pose_raw", String, self.grasp_pose_cb, queue_size=10)
        rospy.Subscriber("/joint_states_single", JointState, self.visual_grasp_pub)
        self.joint_pub = rospy.Publisher("/joint_states_hggd", JointState, queue_size=10)

        # gripper close/open service
        rospy.wait_for_service('/gripper_srv')

        # TF trans
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.width = 0

        # arm pose to piper
        self.arm_pose_pub = rospy.Publisher("/pos_cmd", PosCmd, queue_size=1)

### HGGD pose process
    def visual_grasp_pub(self, msg):
        """
        Discribe:
            Add visual_joint7&8 to jointstate and publish to /joint_states_hggd, then robotstatepublsher publish tf tree, for visualize expect gripper pose later
        
        Args:
            msg: /joint_states_single, realtime 6 joint angle from piper SDK
        """
        # get joint_states_single and add visual joint
        merged = JointState()
        merged.header.stamp = rospy.Time.now()
        merged.name = msg.name + ["visual_joint7", "visual_joint8"]
        merged.position = msg.position + (self.width/2.0, -self.width/2.0)
        merged.velocity = msg.velocity + (0.0, 0.0)
        merged.effort = msg.effort + (0.0, 0.0)
        self.joint_pub.publish(merged)

    def grasp_pose_cb(self, msg: String):
        """
        Discribe:
            Subscribe expect gripper grasp pose from HGGD, transform form camera optical frame to arm base and format to PoseStamped
        
        Args:
            msg: /hggd_pose_raw, translation and rotation json data from HGGD

        Return:
            PoseStamped pose of expect grasp
        """
        self.pose_cb = json.loads(msg.data) # self.pose_cb for get score and width later

        # try the largest score one
        best_grasp = max(self.pose_cb, key=lambda g: g["score"])

        # get translation, rotation and orientation 
        tx, ty, tz = best_grasp["translation"]
        r = best_grasp["rotation"]
        quat = transformations.quaternion_from_matrix([
            [r[0][0], r[0][1], r[0][2], 0],
            [r[1][0], r[1][1], r[1][2], 0],
            [r[2][0], r[2][1], r[2][2], 0],
            [      0,       0,       0, 1]
        ])
        self.width = best_grasp["width"]
        print("width:",self.width)

        # trans to arm base as posestamped
        self.pose = PoseStamped()
        self.pose.header.stamp = rospy.Time.now()
        self.pose.header.frame_id = "camera_color_optical_frame"
        self.pose.pose.position.x = tx
        self.pose.pose.position.y = ty
        self.pose.pose.position.z = tz
        self.pose.pose.orientation.x = quat[0]
        self.pose.pose.orientation.y = quat[1]
        self.pose.pose.orientation.z = quat[2]
        self.pose.pose.orientation.w = quat[3]

        self.pose_armbase = self.tf_buffer.transform(self.pose, "arm_base", rospy.Duration(1.0))
        return self.pose_armbase
###


### publish desire TFs (grasp pos + assist pos)
    def posrot_trans(
            self, 
            father_farme: str, 
            child_frame: str, 
            translation: Union[np.ndarray, list],
            orientation: Union[np.ndarray, list]
            ):
        """
        Discribe:
            Convert a position and rotation vector to a TransformStamped message for broadcasting to the TF tree

        Args:
            father_frame: the name of the father frame of the transformation
            child_frame: the name of the child frame of the transformation
            translation: the translation vector of the child frame relative to the father frame
            orientation: the rotation vector (axis-angle) of the child frame relative to the father frame

        Returns:
            a TransformStamped message representing the transformation from the father frame to the child frame
        """
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = father_farme
        t.child_frame_id = child_frame

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        # axis-angle(direction cosine) to quaternion
        q = SR.from_rotvec(orientation).as_quat()

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        return t
    # creat relative frame according grasp_base_visual frame
    def grasp_frames_transform(self, backdistance: float) -> None:
        """
        Discribe:
            Create expect grasp pose frame grasp_base_visual and static assist grasp frames according grasp_base_visual

        Args:
            backdistance: distance from near to object to actual object(ready to grasp) 
        """
        d = backdistance

        # send tf to grasp base visual, when robot state publisher launched, the link call "grasp base visual" will be correlated
        visual_grasp = geometry_msgs.msg.TransformStamped()
        visual_grasp.header.stamp = rospy.Time.now()
        visual_grasp.header.frame_id = "arm_base"
        visual_grasp.child_frame_id = "grasp_base_visual"
        visual_grasp.transform.translation.x = self.pose_armbase.pose.position.x
        visual_grasp.transform.translation.y = self.pose_armbase.pose.position.y
        visual_grasp.transform.translation.z = self.pose_armbase.pose.position.z
        visual_grasp.transform.rotation.x = self.pose_armbase.pose.orientation.x
        visual_grasp.transform.rotation.y = self.pose_armbase.pose.orientation.y
        visual_grasp.transform.rotation.z = self.pose_armbase.pose.orientation.z
        visual_grasp.transform.rotation.w = self.pose_armbase.pose.orientation.w
        self.tf_broadcaster.sendTransform(visual_grasp)

        rospy.sleep(0.2)

        # quat to roatiton, then extract x axis cosine as gripper orientation
        rotation = SR.from_quat(
            self.pose_armbase.pose.orientation.x,
            self.pose_armbase.pose.orientation.y,
            self.pose_armbase.pose.orientation.z,
            self.pose_armbase.pose.orientation.w,
        ).as_matrix()
        vect = rotation[:, 0]

        # build grasp frame

        trans_origin2backward = self.posrot_trans(
            father_farme = "grasp_base_visual",
            child_frame = "visual_backward",
            translation = [-d*vect[0], -d*vect[1], -d*vect[2]],
            orientation = [0.0, 0.0, 0.0],
        )

        self.tf_broadcaster.sendTransform(
            [
                trans_origin2backward,
            ]
        )
        rospy.sleep(0.2)
###

### target pos
    def fetch_tf_posequat(self, target_frame: str, source_frame: str, wait_block: bool = True):
        """
        Discribe:
            Get transformation from source_frame to target_frame, 
            for pipline to execution, send gripper target pose one by one
        
        Args:
            target_frame: the name of the target frame
            source_frame: the name of the source frame
            wait_block: whether to block and wait for the transformation to be available
        """
        loop_rate_rece = rospy.Rate(2)
        while not rospy.is_shutdown():
            try:
                trans_listener = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time())
                return trans_listener
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):  # type: ignore
                rospy.logwarn(
                    "Fail to fetch transformation... from {} to {}".format(target_frame, source_frame)
                )
                if wait_block is False:
                    return None

                loop_rate_rece.sleep()
                continue

    def fetch_tf_poscmd(
            self,
            target_frame: str,
            source_frame: str,
            wait_block: bool = True,
    ):
        """
        Discribe:
            Get Transformation from source_frame to target_frame, and trans to PosCmd for arm_servo_pose to execute, list translation + euler
        
        Args:
            target_frame: the name of the target frame
            source_frame: the name of the source frame
            wait_block: whether to block and wait for the transformation to be available
        """
        trans_msg = self.fetch_tf_posequat(target_frame, source_frame, wait_block)
        translation = [
            trans_msg.transform.translation.x,
            trans_msg.transform.translation.y,
            trans_msg.transform.translation.z,
        ]
        quat = [
            trans_msg.transform.rotation.x,
            trans_msg.transform.rotation.y,
            trans_msg.transform.rotation.z,
            trans_msg.transform.rotation.w,
        ]
        euler = transformations.euler_from_quaternion(quat)
        pos_cmd = translation + euler
        return pos_cmd
###
 
### execute pos/gripper
    def arm_servo_pose(self, act_pose: list, repeat=False):
        """
        Discribe:
            Arm absolute pose publish
        
        :param self: 说明
        :param act_pose: 说明
        """
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
        if repeat:
            rospy.sleep(0.1)
            self.arm_pose_pub.publish(pos_data)      
    
    # close/open gripper service from piper
    def call_gripper_service(gripper_opening, force_effort=2):
        """
        Discribe:
            Gripper close/open service from arm piper SDK
        
        Args:
            gripper_opening: gripper opening distance, range: 0 - 0.07 m
            force_effort: grasp force, range: 0.5 - 2 N
        """
        global gripper_service
        try:
            gri_cmd = GripperRequest()
            gri_cmd.gripper_angle = gripper_opening
            gri_cmd.gripper_effort = force_effort
            gri_cmd.set_zero = 0
            gri_cmd.gripper_code = 1
            response = gripper_service(gri_cmd)
            rospy.loginfo("Gripper service called successfully. Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e) 
###

### pipline
    def pipline(self):
        # pos from tf
        pose_endposupper = self.fetch_tf_posequat("base", "visual_backward")
        pose_objgrip = self.fetch_tf_posequat("base", "grasp_base_visual")
        # gripper
        grasp_open = 0.07
        grasp_close = self.width
        grasp_ready = min(self.width + 0.02, grasp_open)

        action_flow = [
            {"name": "UR_TO_obj_upper", "type": "APOSE", "value": pose_endposupper},
            {"name": "GRIPPER_INIT", "type": "GRASP", "value": grasp_ready},
            {"name": "UR_TO_pose_grip", "type": "APOSE", "value": pose_objgrip},
            {"name": "GRIPPER_CLOSE", "type": "GRASP", "value": grasp_close},
            {"name": "UR_RE_pose_upper", "type": "APOSE", "value": pose_endposupper},
        ]
        print("Excute action list:", action_flow)

        ex_ret = excute_graspactions(action_flow)
        if not ex_ret:
            if not input_signal("## Interrupt Trial [{}]: [COUNT]:enter, [RETURN]:nine ##".format(trial_id)):
                continue
        trial_id += 1
        if input_signal("## Trial [{}]: [SUCCESS]:enter, [FAIL]:nine ##".format(trial_id)):
            succ_count += 1

    rospy.loginfo("[Pipeline]: Everything done, spinning...")
    rospy.spin()


    def excute_graspactions(action_flow: list, action_prefix: str="test"):
        """execution information output and execution

        Args:
            action_flow: multiple action list
            gripper_client: gripper node
            pose_node: node of robot arm
            action_prefix: description of action type

        Returns:
            execution state
        """
        for action in action_flow:
            if not input_signal(
                "[{}]: ".format(action_prefix)
                + "\n"
                + action["type"]
                + " "
                + action["name"]
                + " {} Execute ? :".format(action["value"])
            ):
                return False
            if action["type"] == "GRASP":
                call_gripper_service(action["value"])
            if action["type"] in ["APOSE", "AJOINT"]:
                arm_servo_pose(action["value"], repeat=True)
        return True

### user
def input_signal(reminder: str = "Enter to contine", supple_txt: str = "[0:End, 9:Retn] "):
    """Control function for executing one step

    Args:
        reminder: description of step. Defaults to "Enter to contine".
        supple_txt: supplementary text to display before the reminder. Defaults to "[0:End, 9:Retn] ".

    Returns:
        whether to continue the execution flow, False if return 9, True if return other keys except 0, and exit if return 0.
    """
    a = input(supple_txt + reminder)
    if a == "0":
        rospy.signal_shutdown("Key interrupt down.")
        rospy.sleep(0.2)
        exit(0)
    return a != "9"

    




if __name__ == '__main__':
    hggd = HGGDCallback()
    rospy.spin()