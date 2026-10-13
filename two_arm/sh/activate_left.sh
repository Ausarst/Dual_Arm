source devel/setup.bash
bash can_activate.sh can0 1000000 1-3:1.0 # use "sudo ethtool -i can0" if you don't know the last param
roslaunch piper all_start_single_piper.launch can_port:=can0 auto_enable:=true

