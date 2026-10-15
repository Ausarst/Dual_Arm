#!/usr/bin/env python3
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

class HGGDCallback:
    def __init__(self):
        rospy.init_node("hggd_pose_process")
        rospy.Subscriber("/hggd_pose_raw", String, self.grasp_pose_cb, queue_size=10)
        # self.pub = rospy.Publisher("/joint_states_single", JointState, queue_size=10)
        # self.pub = rospy.Publisher("/hggd_grasp_pose", PoseStamped, queue_size=10)
        # self.marker_pub = rospy.Publisher("/visual_grasp_marker", Marker, queue_size=10)
        # self.pub6 = rospy.Publisher("/gripper_link6", Marker, queue_size=10)
        # self.pub7 = rospy.Publisher("/gripper_link7", Marker, queue_size=10)
        # self.pub8 = rospy.Publisher("/gripper_link8", Marker, queue_size=10)
        rospy.Subscriber("/joint_states_single", JointState, self.visual_grasp_pub)
        self.joint_pub = rospy.Publisher("/joint_states_hggd", JointState, queue_size=10)

        # TF trans
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
    

    def grasp_pose_cb(self, msg: String):
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

        # trans to posestamped
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

        self.pose_ready = True # make sure hggd pose is ok
        print("ready",self.pose_ready)


    
    def visual_grasp_pub(self, msg):

        if not hasattr(self, "pose_ready") or not self.pose_ready:
            return

        # send tf to grasp base visual link
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "arm_base"
        t.child_frame_id = "grasp_base_visual"
        t.transform.translation.x = self.pose_armbase.pose.position.x
        t.transform.translation.y = self.pose_armbase.pose.position.y
        t.transform.translation.z = self.pose_armbase.pose.position.z
        t.transform.rotation.x = self.pose_armbase.pose.orientation.x
        t.transform.rotation.y = self.pose_armbase.pose.orientation.y
        t.transform.rotation.z = self.pose_armbase.pose.orientation.z
        t.transform.rotation.w = self.pose_armbase.pose.orientation.w
        self.tf_broadcaster.sendTransform(t)

        # send grasp open width

        # js = JointState()
        # js.header.stamp = rospy.Time.now()
        # js.name = ["visual_joint7", "visual_joint8"]
        # js.position = [self.width/2.0, -self.width/2.0] # m
        # self.joint_pub.publish(js)

        # TODO: since robot_state_publisher can only subscribe one topic, we need add the visual_joint7 and visual_joint8 to /joint_states_single then publish to /joint_states
    
        # get joint_states_single and add visual joint
        merged = JointState()
        merged.header.stamp = rospy.Time.now()
        merged.name = msg.name + ["visual_joint7", "visual_joint8"]
        merged.position = msg.position + [self.width/2.0, -self.width/2.0]
        merged.velocity = msg.velocity + [0.0, 0.0]
        merged.effort = msg.effort + [0.0, 0.0]
        self.joint_pub.publish(merged)

if __name__ == '__main__':
    # try:
    #     hggd = HGGDCallback()
    #     while not rospy.is_shutdown():
    #         hggd.visual_grasp_pub()
    # except rospy.ROSInterruptException:
    #     pass
    hggd = HGGDCallback()
    rospy.spin()