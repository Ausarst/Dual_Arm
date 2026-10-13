import rospy
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String
from cv_bridge import CvBridge
# import cv2
# import message_filters
from sensor_msgs.msg import CameraInfo
import numpy as np
import json
bridge = CvBridge()

class GetCamInfo:
    def __init__(self):
        self.d435_intrinsics = None 
        self.get_depth = None 
        self.get_rgb = None
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", ROSImage, self.get_depth_image) 
        # rospy.Subscriber("/camera/depth/image_rect_raw", ROSImage, get_depth_image) 
        rospy.Subscriber("/camera/color/image_raw", ROSImage, self.get_rgb_image)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.get_d435_intrinsics)
        
    def get_d435_intrinsics(self, cam_info):
        self.cam_info = cam_info
        self.d435_intrinsics = np.array(self.cam_info.K).reshape(3, 3)

    def get_depth_image(self, depth_msg):
        self.depth_msg = depth_msg
        self.get_depth = bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
        # get_depth = cv2.resize(depth_image, (1280, 720))

    def get_rgb_image(self, rgb_msg):
        self.rgb_msg = rgb_msg
        self.get_rgb = bridge.imgmsg_to_cv2(self.rgb_msg, desired_encoding="bgr8") / 255.0
        # get_rgb = cv2.resize(rgb_image, (1280, 720)) / 255.0

    def cam_intrinsics(self):
        return self.d435_intrinsics
    
    def cam_images(self):
        return self.get_depth, self.get_rgb


class GraspPose:
    def __init__(self):
        # Subscribe camera topices and Publish grasp pose
        # rospy.init_node('grasp_inference_node', anonymous=True) 
        self.pub = rospy.Publisher('/hggd_pose_raw', String, queue_size=10)

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

# while using a script class to import, you can int the node in the main script.