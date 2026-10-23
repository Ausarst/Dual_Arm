#!/usr/bin/env python3
""" 零点写入应答校验工具 (诊断"标零后上电复位"问题)

    官方 piper_set_joint_zero.py 发送 JointConfig(set_zero=0xAE) 后从不检查固件应答,
    本工具补上这一步:
        1. ClearRespSetInstruction 清空SDK缓存的应答
        2. 发送 JointConfig(motor, 0xAE)  (不移动机械臂, 仅写零指令)
        3. 轮询 GetRespInstruction() 读固件应答 is_set_zero_successfully

    结果解读:
        is_set_zero_successfully == 1  : 固件确认零点已写入
            -> 断电重启后若零位仍复位为上电位形, 则为固件持久化/硬件问题, 带证据联系agilex
        is_set_zero_successfully == 0  : 固件明确拒绝写入
            -> 尝试不同电机使能状态(先EnableArm/先DisableArm)重发, 仍拒绝则疑固件版本过旧
        一直为 -1 (无应答)             : 0x475 帧被丢弃或应答帧未上报, 疑固件版本不匹配

    用法 (先停掉占用CAN的节点):
        python3 zero_verify.py can0 7        # 对can0臂的全部电机(7)做写入+查应答
        python3 zero_verify.py can0 1        # 只对1号电机
        python3 zero_verify.py can0 r        # 只读当前应答缓存, 不发送
"""

import sys
import time
from piper_sdk import C_PiperInterface_V2


def read_resp(piper, wait=2.0):
    t0 = time.time()
    while time.time() - t0 < wait:
        resp = piper.GetRespInstruction()
        r = resp.instruction_response
        if r.is_set_zero_successfully != -1:
            return r
        time.sleep(0.05)
    return piper.GetRespInstruction().instruction_response


def main():
    can = sys.argv[1] if len(sys.argv) > 1 else "can0"
    what = sys.argv[2] if len(sys.argv) > 2 else "r"

    piper = C_PiperInterface_V2(can_name=can)
    piper.ConnectPort()
    time.sleep(0.3)

    if what == "r":
        r = read_resp(piper, wait=1.0)
        print("当前应答缓存: instruction_index=%s is_set_zero_successfully=%s" %
              (r.instruction_index, r.is_set_zero_successfully))
        return

    motor = int(what)
    print("清空应答缓存...")
    piper.ClearRespSetInstruction()

    print("发送 JointConfig(motor=%d, set_zero=0xAE) (不移动机械臂)..." % motor)
    piper.JointConfig(joint_num=motor, set_zero=0xAE)

    r = read_resp(piper)
    print("固件应答: instruction_index=%s" % r.instruction_index)
    v = r.is_set_zero_successfully
    if v == 1:
        print("is_set_zero_successfully = 1  -> 固件确认零点写入成功")
        print("下一步: 断电重启验证持久性; 若仍复位 -> 固件持久化/硬件问题, 联系agilex")
    elif v == 0:
        print("is_set_zero_successfully = 0  -> 固件明确拒绝写入")
        print("可尝试: EnableArm(motor)后再发 / DisableArm(motor)后再发, 仍拒绝则疑固件过旧")
    else:
        print("is_set_zero_successfully = %s (无应答) -> 0x475被丢弃或应答未上报, 疑固件版本不匹配" % v)


if __name__ == '__main__':
    main()
