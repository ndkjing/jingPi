"""
多线程同时控制
"""


from module.wheel_module import WheelModule
from module.ultrasonic_module import UltrasonicModule
# from module.pid_module import PIDModule

import numpy as np
from collections import deque
import platform
import threading, signal
import cv2
import math
import time
import  sys


# 生成模块控制类
camera= WheelModule()  # 等待修改
wheel = WheelModule(11, 12, 13, 15)

forward_ultrasonic = UltrasonicModule(32, 36)
left_ultrasonic = UltrasonicModule(38, 40)
#TODO 通过pid控制转动使得distance_x,distance_y均趋近0 保持人在中间


#  用于控制运动的全局变量
person_1 = []   # 人坐标
forward_ultrasonic_distance = 0     # 前面超声波检测距离
left_ultrasonic_distance = 0     # 左侧超声波检测距离

# 视频的宽和长
w=0
h=0

#控制运动状态
status='forward'

# pid初始化参数
kp=0.09
ki=0.08
kd=0.002
# 根据相机居中人物pid输出
camera_center_pid_output = 0

# 超声波检测时间间隔
ultrasonic_detect_interval = 0.05
#摄像头检测时间间隔(单位：秒)
get_fream_interval = 1.5
# 获取视频流
vc = cv2.VideoCapture(0)

c=1  #帧数
#获取视频FPS,使任何fps速度视频均按2S取帧
# 查看OpenCV版本信息
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = vc.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
timeF = math.ceil(fps) * get_fream_interval



def get_person_position():
    global c
    global person_1
    global w,h
    while True:
        rval , fream = vc.read()
        h, w = fream.shape[:2]
        # print(rval)
        if rval is False:
            print('video is over')
            break

        # 每隔指定间隔检测是否有人出现在视野中
        if(c%timeF==0):
            # 人物中心到图片中心距离  返回人物框[人物中心x轴,中心y轴,框宽w,高度h]
            starTime = time.time()
            persons = camera.get_person_center(fream)
            if len(persons) > 0:  # 当检测到人时仅返回第一人位置因为只能跟踪一个人
                person_1 = persons[0]
                print(person_1)
            endTime = time.time()
            print('detect fps:',1/(endTime-starTime))
        #  TODO 以检测间隔的1/10 执行追踪  tracker有问题
        # elif(c%int(timeF/10))==0:
        #     persons = camera.track_person(fream)
        #     if len(persons) > 0:
        #         person_1 = persons[0]
        #         print(person_1)

        c=c+1


def get_ultrasonic_distance():
    global ultrasonic_detect_interval
    global forward_ultrasonic_distance,left_ultrasonic_distance
    while True:
        forward_ultrasonic_distance = forward_ultrasonic.getDistance()
        left_ultrasonic_distance = left_ultrasonic.getDistance()
        print('前面超声波距离(cm)：',forward_ultrasonic_distance)
        print('左侧超声波距离(cm)：',left_ultrasonic_distance)
        time.sleep(ultrasonic_detect_interval)


def wheel_status():
    global person_1
    global left_ultrasonic_distance,forward_ultrasonic_distance
    global w,h
    global status
    global lock
    while True:
        if len(person_1)>0:
            pixel_distance_x = person_1[0] - w / 2  # 人中心到图像中心距离
            pixel_distance_y = person_1[1] - h / 2
            # distance_x,distance_y符合以图像中点构建坐标系的正负坐标表示
            print(pixel_distance_x, pixel_distance_y)
            width_error = pixel_distance_x / w   # 到中间距离与宽度的比例
            if width_error > 0.05 and left_ultrasonic_distance >30:  # 误差大于5%   转向
                status = 'right'
            elif width_error < -0.1:
                status = 'left'
        if forward_ultrasonic_distance >= 40 :  # 小于40厘米停车
            lock.acquire()
            status='forward'
            lock.release()
        elif forward_ultrasonic_distance < 20 or forward_ultrasonic_distance>1000:
            lock.acquire()
            status = 'backward'
            lock.release()
        elif forward_ultrasonic_distance < 40 and left_ultrasonic_distance > 30:

            status = 'left'

        elif forward_ultrasonic_distance < 40 and left_ultrasonic_distance <= 30:

            status = 'backward'

        else:

            status='right'

        time.sleep(0.2)


def pid_process():
    global kp,ki,kd
    global camera_center_pid_output
    global person_1
    global w
    # create a PID and initialize it
    p = PIDModule(kp, ki, kd)
    p.initialize()

    # loop indefinitely
    while True:
        if len(person_1)>0:
            pixel_distance_x = person_1[0] - w / 2  # 人中心到图像中心距离
            pixel_distance_y = person_1[1] - h / 2
            # distance_x,distance_y符合以图像中点构建坐标系的正负坐标表示
            camera_center_pid_output = p.update(pixel_distance_x)  # 输入误差执行pid算法 返回误差调控 用于控制速度


def wheel_control():
    global status
    global person_1
    global camera_center_pid_output
    global wheel
    while True:
        time.sleep(0.5)
        print(status)
        if status =='forward':
            wheel.forward()
            time.sleep(0.5)
            # wheel.stop()
        elif status =='backward':

            wheel.backward()
            time.sleep(0.5)
            # wheel.stop()
        elif status =='left'and len(person_1)>0:  # 若是因为检测到人而转向使用pid控制转向时间
            wheel.left()
            time.sleep(0.5)
            # wheel.stop()
        elif status =='left': # 因为障碍物而转向
            wheel.left()
            time.sleep(0.5)
            # wheel.stop()
        # elif status =='right'and len(person_1)>0:  # 若是因为检测到人而转向使用pid控制转向时间
        #     wheel.right()
        #     time.sleep(0.5)
        #     # wheel.stop()
        elif status =='right': # 因为障碍物而转向
            wheel.right()
            time.sleep(0.5)
            # wheel.stop()
        elif status == 'stop':
            # wheel.backward()
            # time.sleep(0.5)
            wheel.stop()
            time.sleep(0.5)

def quit(signum,frame):
    print('quit')
    wheel.quit()
    vc.release()
    sys.exit()

lock = threading.Lock()  # 用threading.Lock()产用生一把锁

signal.signal(signal.SIGINT, quit)
signal.signal(signal.SIGTERM, quit)

wheel_thread = threading.Thread(target=wheel_control)

status_thread = threading.Thread(target=wheel_status)

ultrasonic_thread = threading.Thread(target=get_ultrasonic_distance)

camera_thread = threading.Thread(target=get_person_position)



wheel_thread.start()
status_thread.start()
ultrasonic_thread.start()
camera_thread.start()


status_thread.setDaemon(True)
wheel_thread.setDaemon(True)
ultrasonic_thread.setDaemon(True)
camera_thread.setDaemon(True)

while True:
    pass
# status_thread.join()
# wheel_thread.join()
# ultrasonic_thread.join()
# camera_thread.join()




