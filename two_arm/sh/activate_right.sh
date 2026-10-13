source devel/setup.bash
bash can_activate.sh can1 1000000 1-2:1.0 # use "sudo ethtool -i can0" if you don't know the last param
roslaunch piper all_start_single_piper.launch can_port:=can1 auto_enable:=true

