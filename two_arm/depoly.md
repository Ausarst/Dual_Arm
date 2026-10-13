### Arm control

Activate the arm and display in rviz
```bash
cd ~/two_arm
source devel/setup.bash
sudo ethtool -i can0 # eg.can0 bus 1-2:1.0
bash can_activate.sh can0 1000000 1-2:1.0 
roslaunch piper piper_realtime_rviz.launch  can_port:=can0 auto_enable:=true
```
activate the camera (d405)
```bash
roslaunch realsense2_camera rs_camera.launch enable_pointcloud:=true
```

publish tf from camera to arm link
```bash
rosrun tf2_ros static_transform_publisher \ -0.06 0 0.06 0 0.707 0 -0.707 link6 camera_link
```

d435
```bash
roslaunch realsense2_camera rs_camera_hggd.launch 
```
```bash
rosrun tf2_ros static_transform_publisher -0.055 0 0.06 0 0.707 0 -0.707 link6 camera_link
```

shot down the arm
```bash
rosservice call /go_zero_srv "is_mit_mode: false"
rosservice call /reset_srv # hold the arm
```

be sure controller is PS3/PC Gamepad
```bash
roslaunch piper rt_armpose.launch 
[INFO] [1768463013.536321081]: Opened joystick: /dev/input/js0 (ShanWan PS3/PC Gamepad). deadzone_: 0.050000.
```

```bash
smart@smart:~/code/HGGD$ ca
(base) smart@smart:~/code/HGGD$ ca hggd
(hggd) smart@smart:~/code/HGGD$ bash demo.sh
```

view TF
```bash
rosrun tf view_frames
```

---

Activate camera d435 first

Activate the HDDG to count the grasp pose, it will publish `/grasp_pose` rostopic as a json message, main script is demo_serv.py
```bash
source /opt/ros/noetic/setup.bash
conda activate hggd_env
bash demo.sh
```

Service for image inference
```bash
rosservice call /trigger_inference
```

Subscribe pose topic
```bash
rostopic echo /grasp_pose
```


```bash
smart@smart:~/two_arm/src/piper/scripts$ python3 hggd_endpose.py
```
TODO: publish grasp pose calculated by HGGD to rviz to visualize

Update:
From `~/two_arm/src/piper/scripts`, receive arm joint callback `/joint_states_single` and hggd calculated pose `/hggd_pose_raw`, then publish `/joint_states_hggd`, include original callback and visual grasp endeffector pose
```bash
python3 hggd_endpose.py
```

**arm_with_camera.urdf.xacro**: add camera model to arm in rviz\
piper model: ``$(find piper_description)/urdf/piper_description.xacro``\
camera model: ``$(find realsense2_description)/urdf/_d435.urdf.xacro``\
TF: ``xyz="-0.055 0 0.06" rpy="0 -1.5708 0"``\
**arm_camera_rviz.launch**: rviz launch file
```bash
roslaunch piper arm_camera_rviz.launch 
```

---
#### Multiple arm

can0(up one) left arm, can1(down one) right arm
```bash
cd ~/two_arm
source devel/setup.bash
sudo ethtool -i can0
sudo ethtool -i can1
bash can_activate.sh can0 1000000 1-3:1.0
bash can_activate.sh can1 1000000 1-2:1.0
```

```bash
roslaunch piper 2arm_camera_rviz.launch arm1_can_port:=can0 arm2_can_port:=can1
```

### Data stream

---

**This is for joint state Translation + Euler**

**src/piper/scripts/joy2delta_twist_cmds.py** \
Subscribe `/joy` from controllor, publish `servo_server/delta_twist_cmds` joint and `/joy_relat_posecmd` endpose \
-> **src/piper/scripts/agile_armpose.py** \
Subscribe `/joy_relat_posecmd`, sum the arm current pose subcribe from `/end_pose` published by piper_ctrl_single_node.py, producing absolute pose publish to `/pos_cmd` \

```python
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
```

-> **src/piper/scripts/piper_ctrl_single_node.py** \
Subscribe `/pos_cmd`, producing by `pos_callback()`

```py
    def pos_callback(self, pos_data):
        """机械臂末端位姿订阅回调函数

        Args:
            pos_data (): 
        """
        # rospy.loginfo("Received PosCmd:")
        # rospy.loginfo("x: %f", pos_data.x)
        # rospy.loginfo("y: %f", pos_data.y)
        # rospy.loginfo("z: %f", pos_data.z)
        # rospy.loginfo("roll: %f", pos_data.roll)
        # rospy.loginfo("pitch: %f", pos_data.pitch)
        # rospy.loginfo("yaw: %f", pos_data.yaw)
        # rospy.loginfo("gripper: %f", pos_data.gripper)
        # rospy.loginfo("mode1: %d", pos_data.mode1)
        # rospy.loginfo("mode2: %d", pos_data.mode2)
        if not self.block_ctrl_flag:
            factor = 180 / 3.1415926
            x = round(pos_data.x*1000) * 1000
            y = round(pos_data.y*1000) * 1000
            z = round(pos_data.z*1000) * 1000
            rx = round(pos_data.roll*1000*factor) 
            ry = round(pos_data.pitch*1000*factor)
            rz = round(pos_data.yaw*1000*factor)
            rospy.loginfo("Received PosCmd:")
            rospy.loginfo("x: %f", x)
            rospy.loginfo("y: %f", y)
            rospy.loginfo("z: %f", z)
            rospy.loginfo("roll: %f", rx)
            rospy.loginfo("pitch: %f", ry)
            rospy.loginfo("yaw: %f", rz)
            rospy.loginfo("gripper: %f", pos_data.gripper)
            rospy.loginfo("mode1: %d", pos_data.mode1)
            rospy.loginfo("mode2: %d", pos_data.mode2)
            if(self.GetEnableFlag()):
                self.piper.MotionCtrl_1(0x00, 0x00, 0x00)
                self.piper.MotionCtrl_2(0x01, 0x00, 50)
                self.piper.EndPoseCtrl(x, y, z, 
                                        rx, ry, rz)
                gripper = round(pos_data.gripper*1000*1000)
                if(pos_data.gripper>80000): gripper = 80000
                if(pos_data.gripper<0): gripper = 0
                if self.gripper_exist and pos_data.gripper != -1:
                    # self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0) # MDF
                    self.piper.GripperCtrl(abs(gripper), 3000, 0x01, 0)
                self.piper.MotionCtrl_2(0x01, 0x00, 50)
```

self.piper.EndPoseCtrl(x, y, z, rx, ry, rz) from piper sdk for IK calculation



For ROS script, figue out the message stream of publisher & subcriber rostopic then you will understand whole process \ 
Subcriber -> find the callback function;\ 
Publisher -> find who subcribe this rostopic