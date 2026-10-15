#!/usr/bin/env python3
""" Dual arm grasp pose process & execution (per arm instance)

    Usage:
        rosrun piper hggd_endpose_dual_arm.py _arm_ns:=arm1
        (arm2 same with _arm_ns:=arm2, camera currently mounted on arm1)

    Subscribe:
        /hggd_pose_raw                     HGGD grasp pose json, in camera optical frame
        /{arm_ns}/joint_states_single      joint feedback from piper SDK

    Publish:
        /{arm_ns}/joint_states_hggd        joint states + virtual gripper joints for rviz
        /{arm_ns}/pos_cmd                  cartesian pose cmd to piper SDK (x/y/z m, rpy rad)

    TF broadcast (all frames prefixed by arm_ns, 10Hz timer republish,
                  so relay frames never expire and target switch refreshes automatically):
        {arm_ns}/grasp_base_visual         expected grasp pose in arm base,
                                           same name as URDF fixed frame to drive the ghost gripper STL
        {arm_ns}/visual_backward           relay frame, retreat along grasp approach (-x) axis

    Service:
        /{arm_ns}/gripper_srv              gripper open width, unit: meter [0, 0.07]

    Pipeline: numeric pose composition (no TF round trip for execution),
              each action confirmed by keyboard before executing (input_signal).
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
from sensor_msgs.msg import JointState

from typing import Union
import numpy as np
from scipy.spatial.transform import Rotation as SR

from piper_msgs.srv import Gripper, GripperRequest
from piper_msgs.msg import PosCmd


class HGGDCallback:
    def __init__(self):
        # init first: CLI _arm_ns:=x is uploaded to the param server by init_node,
        # launch <param> is written there beforehand, so both paths converge
        # into the same get_param call below
        rospy.init_node("hggd_pose_process", anonymous=True)
        # which arm this instance controls, all topics / frames / services prefixed by it
        self.arm_ns = rospy.get_param("~arm_ns", "arm1")

        # camera frame the HGGD pose defined in:
        # default use the URDF clone frame published by robot_state_publisher (on /tf_static),
        # switch to driver frame (camera_color_optical_frame) once link6->camera_link bridge TF added
        self.cam_frame = rospy.get_param(
            "~cam_frame", "%s/camera_color_optical_frame" % self.arm_ns)
        self.arm_base_frame = "%s/arm_base" % self.arm_ns

        rospy.Subscriber("/hggd_pose_raw", String, self.grasp_pose_cb, queue_size=10)
        rospy.Subscriber("/%s/joint_states_single" % self.arm_ns,
                         JointState, self.visual_grasp_pub)
        self.joint_pub = rospy.Publisher(
            "/%s/joint_states_hggd" % self.arm_ns, JointState, queue_size=10)

        # arm pose to piper
        self.arm_pose_pub = rospy.Publisher(
            "/%s/pos_cmd" % self.arm_ns, PosCmd, queue_size=1)

        # gripper close/open service
        rospy.wait_for_service("/%s/gripper_srv" % self.arm_ns)
        self.gripper_service = rospy.ServiceProxy(
            "/%s/gripper_srv" % self.arm_ns, Gripper)

        # TF trans
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.width = 0.0
        self.pose_armbase = None  # PoseStamped of expected grasp in arm base
        self.pose_ready = False   # set by hggd callback, consumed by main loop pipeline

        # relay distance from grasp pose along approach axis
        self.relay_backdist = rospy.get_param("~relay_backdist", 0.10)

        # conservative workspace check bounds in arm base frame (m), tune per piper spec
        self.ws_r_min = rospy.get_param("~ws_r_min", 0.10)
        self.ws_r_max = rospy.get_param("~ws_r_max", 0.60)
        self.ws_z_min = rospy.get_param("~ws_z_min", 0.00)
        self.ws_z_max = rospy.get_param("~ws_z_max", 0.50)

        # republish grasp & relay frames at 10Hz: keep TF cache fresh for rviz,
        # and new HGGD pose is picked up automatically on next tick
        self.tf_timer = rospy.Timer(rospy.Duration(0.1), self.broadcast_grasp_frames)

### HGGD pose process
    def visual_grasp_pub(self, msg):
        """
        Discribe:
            Add visual_joint7&8 to jointstate and publish to /{arm_ns}/joint_states_hggd,
            then robotstatepublsher publish tf tree, for visualize expect gripper pose later

        Args:
            msg: /{arm_ns}/joint_states_single, realtime 6 joint angle from piper SDK
        """
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
            Subscribe expect gripper grasp pose from HGGD, pick the best score one,
            transform from camera optical frame to arm base frame and store for pipeline

        Args:
            msg: /hggd_pose_raw, translation and rotation json data from HGGD
        """
        grasps = json.loads(msg.data)
        if not grasps:
            rospy.logwarn("HGGD returned empty grasp list, skip")
            return

        best_grasp = max(grasps, key=lambda g: g["score"])

        tx, ty, tz = best_grasp["translation"]
        r = best_grasp["rotation"]
        quat = transformations.quaternion_from_matrix([
            [r[0][0], r[0][1], r[0][2], 0],
            [r[1][0], r[1][1], r[1][2], 0],
            [r[2][0], r[2][1], r[2][2], 0],
            [      0,       0,       0, 1]
        ])
        self.width = best_grasp["width"]
        rospy.loginfo("grasp score: %.3f width: %.4f", best_grasp["score"], self.width)

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.cam_frame
        pose.pose.position.x = tx
        pose.pose.position.y = ty
        pose.pose.position.z = tz
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        try:
            self.pose_armbase = self.tf_buffer.transform(
                pose, self.arm_base_frame, rospy.Duration(1.0))
        except tf2_ros.TransformException as e:
            rospy.logerr("transform %s -> %s failed: %s", self.cam_frame, self.arm_base_frame, e)
            return
        self.pose_ready = True
###

### TF broadcast (timer driven, for rviz visualization only)
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

    def broadcast_grasp_frames(self, event=None):
        """
        Discribe:
            Timer callback, publish grasp_base_visual (same name as URDF fixed frame,
            so ghost gripper STL follows) and relay frames (visual_backward etc.)
            at 10Hz. Execution does not depend on these, they are for rviz only.
        """
        if self.pose_armbase is None:
            return
        p = self.pose_armbase.pose

        # grasp target frame, same frame name as URDF link so rviz ghost gripper moves
        visual_grasp = geometry_msgs.msg.TransformStamped()
        visual_grasp.header.stamp = rospy.Time.now()
        visual_grasp.header.frame_id = self.arm_base_frame
        visual_grasp.child_frame_id = "%s/grasp_base_visual" % self.arm_ns
        visual_grasp.transform.translation.x = p.position.x
        visual_grasp.transform.translation.y = p.position.y
        visual_grasp.transform.translation.z = p.position.z
        visual_grasp.transform.rotation.x = p.orientation.x
        visual_grasp.transform.rotation.y = p.orientation.y
        visual_grasp.transform.rotation.z = p.orientation.z
        visual_grasp.transform.rotation.w = p.orientation.w

        # relay frame: retreat along grasp approach (-x) axis, for approach start / retreat
        rotation = SR.from_quat(
            [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        ).as_matrix()
        approach = rotation[:, 0]
        d = self.relay_backdist
        trans_origin2backward = self.posrot_trans(
            father_farme="%s/grasp_base_visual" % self.arm_ns,
            child_frame="%s/visual_backward" % self.arm_ns,
            translation=[-d*approach[0], -d*approach[1], -d*approach[2]],
            orientation=[0.0, 0.0, 0.0],
        )

        self.tf_broadcaster.sendTransform([visual_grasp, trans_origin2backward])
###

### target pos (numeric composition, execution never blocked by TF)
    def grasp_poscmd(self, backdist: float = 0.0) -> list:
        """
        Discribe:
            Compose pos_cmd (x y z roll pitch yaw, m & rad) of the grasp pose moved
            backdist along the approach (-x) axis, purely numeric from pose_armbase

        Args:
            backdist: distance to retreat along approach axis, 0 means grasp pose itself
        """
        p = self.pose_armbase.pose
        quat = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        rotation = SR.from_quat(quat).as_matrix()
        approach = rotation[:, 0]
        pos = [
            p.position.x - backdist*approach[0],
            p.position.y - backdist*approach[1],
            p.position.z - backdist*approach[2],
        ]
        euler = transformations.euler_from_quaternion(quat)
        return pos + list(euler)

    def check_workspace(self, name: str, pos_cmd: list) -> bool:
        """
        Discribe:
            Simple sanity check in arm base frame before sending to the real arm,
            reject the action when target out of conservative workspace bounds
        """
        r = np.sqrt(pos_cmd[0]**2 + pos_cmd[1]**2)
        ok = (self.ws_r_min <= r <= self.ws_r_max) and (self.ws_z_min <= pos_cmd[2] <= self.ws_z_max)
        if not ok:
            rospy.logerr("[%s] out of workspace: r=%.3f z=%.3f (bounds r[%.2f,%.2f] z[%.2f,%.2f]), abort",
                         name, r, pos_cmd[2], self.ws_r_min, self.ws_r_max, self.ws_z_min, self.ws_z_max)
        return ok

    def fetch_tf_posequat(self, target_frame: str, source_frame: str, wait_block: bool = True):
        """
        Discribe:
            Debug helper: get transformation from source_frame to target_frame by TF

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
            Debug helper: get transformation from source_frame to target_frame by TF,
            and trans to PosCmd list for arm_servo_pose to execute, translation + euler

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
            Arm absolute pose publish, x/y/z in m, roll/pitch/yaw in rad

        :param act_pose: [x, y, z, roll, pitch, yaw]
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
    def call_gripper_service(self, gripper_opening, force_effort=2):
        """
        Discribe:
            Gripper close/open service from arm piper SDK

        Args:
            gripper_opening: gripper opening distance, range: 0 - 0.07 m
            force_effort: grasp force, range: 0.5 - 2 N
        """
        try:
            gri_cmd = GripperRequest()
            gri_cmd.gripper_angle = gripper_opening
            gri_cmd.gripper_effort = force_effort
            gri_cmd.set_zero = 0
            gri_cmd.gripper_code = 1
            response = self.gripper_service(gri_cmd)
            rospy.loginfo("Gripper service called successfully. Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
###

### pipline
    def pipline(self):
        """
        Discribe:
            Grasp pipeline: numeric pose composition + manual step confirm execution.
            upper(visual_backward) -> open gripper -> grasp pose -> close gripper -> retreat
        """
        if self.pose_armbase is None:
            rospy.logwarn("no grasp pose yet, skip pipeline")
            return

        d = self.relay_backdist
        pose_endposupper = self.grasp_poscmd(backdist=d)
        pose_objgrip = self.grasp_poscmd(backdist=0.0)

        # workspace check before touching the real arm
        if not (self.check_workspace("upper", pose_endposupper)
                and self.check_workspace("grip", pose_objgrip)):
            return

        # gripper
        grasp_open = 0.07
        grasp_close = min(self.width, grasp_open)
        grasp_ready = min(self.width + 0.02, grasp_open)

        action_flow = [
            {"name": "UR_TO_obj_upper", "type": "APOSE", "value": pose_endposupper},
            {"name": "GRIPPER_INIT", "type": "GRASP", "value": grasp_ready},
            {"name": "UR_TO_pose_grip", "type": "APOSE", "value": pose_objgrip},
            {"name": "GRIPPER_CLOSE", "type": "GRASP", "value": grasp_close},
            {"name": "UR_RE_pose_upper", "type": "APOSE", "value": pose_endposupper},
        ]
        rospy.loginfo("Excute action list:")
        for action in action_flow:
            rospy.loginfo("  %s %s: %s", action["type"], action["name"], action["value"])

        self.excute_graspactions(action_flow)

    def excute_graspactions(self, action_flow: list, action_prefix: str="grasp"):
        """
        Discribe:
            Execution information output and execution, each action confirmed by keyboard first

        Args:
            action_flow: multiple action list
            action_prefix: description of action type

        Returns:
            execution state
        """
        for action in action_flow:
            if not safe_input_signal(
                "[{}]: ".format(action_prefix)
                + "\n"
                + action["type"]
                + " "
                + action["name"]
                + " {} Execute ? :".format(action["value"])
            ):
                return False
            if action["type"] == "GRASP":
                self.width = action["value"]  # virtual gripper preview follows pipeline
                self.call_gripper_service(action["value"])
            if action["type"] in ["APOSE", "AJOINT"]:
                self.arm_servo_pose(action["value"], repeat=True)
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


def safe_input_signal(reminder: str = "Enter to contine", supple_txt: str = "[0:End, 9:Retn] "):
    """input_signal guarded against EOF (nohup / closed stdin):
    abort the remaining actions instead of killing the node with EOFError"""
    try:
        return input_signal(reminder, supple_txt)
    except EOFError:
        rospy.logwarn("stdin closed (EOF), abort remaining actions but keep node alive")
        return False


if __name__ == '__main__':
    hggd = HGGDCallback()
    rospy.loginfo("[hggd_endpose_dual_arm] controlling arm: %s, camera frame: %s", hggd.arm_ns, hggd.cam_frame)
    # main thread consumes grasp poses one by one, callbacks run in their own threads
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if hggd.pose_ready:
            hggd.pose_ready = False
            hggd.pipline()
        rate.sleep()
