"""
单线程顺序控制
"""

from module.camera_module import CameraModule
from module.wheel_module import WheelModule
from module.ultrasonic_module import UltrasonicModule
from module.pid_module import PIDModule
from module.control_module import ControlModule
# import numpy as np
# from collections import deque
# import platform
# import threading, signal
import cv2
import math
import time
import  sys


# 生成模块控制类

wheel = WheelModule(11, 12, 13, 15)

forward_ultrasonic = UltrasonicModule(32, 36)
left_ultrasonic = UltrasonicModule(38, 40)

control = ControlModule()
#TODO 通过pid控制转动使得distance_x,distance_y均趋近0 保持人在中间






def get_ultrasonic_distance(control):
    control.forward_ultrasonic_distance = forward_ultrasonic.getDistance()
    control.left_ultrasonic_distance = left_ultrasonic.getDistance()
    print('前面超声波距离(cm)：',control.forward_ultrasonic_distance)
    print('左侧超声波距离(cm)：',control.left_ultrasonic_distance)



def wheel_status(control):

    get_ultrasonic_distance(control)

    if control.forward_ultrasonic_distance >= 40 and control.forward_ultrasonic_distance < 1000:  # 小于40厘米停车
        control.status='forward'

    elif control.forward_ultrasonic_distance < 20 or control.forward_ultrasonic_distance>=1000:
        control.status = 'backward'

    elif control.forward_ultrasonic_distance < 40 and control.left_ultrasonic_distance > 30:
        control.status = 'left'

    elif control.forward_ultrasonic_distance < 40 and control.left_ultrasonic_distance <= 30:
        control.status = 'backward'

    else:
        control.status='right'


try:
    while True:
        wheel_status(control)
        print(control.status)
        # wheel.backward()
        # time.sleep(0.5)
        if control.status== 'forward':
            wheel.forward()
            time.sleep(0.5)
            wheel.stop()
            time.sleep(0.05)
        elif control.status == 'backward':
            wheel.left()
            time.sleep(0.1)
            wheel.backward()
            time.sleep(0.5)
            # wheel.stop()

        elif control.status == 'left':  # 因为障碍物而转向
            wheel.left()
            time.sleep(0.5)

        elif control.status == 'right':  # 因为障碍物而转向
            wheel.right()
            time.sleep(0.5)
            # wheel.stop()
        elif control.status == 'stop':
            wheel.stop()
        # time.sleep(0.2)

except :
    print('quit')
    wheel.quit()






