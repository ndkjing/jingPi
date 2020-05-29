"""
预览树莓派摄像头
"""

import picamera
import picamera.array
import time

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)  # resolution
    camera.framerate = 30  # frame rate
    print("start preview direct from GPU")
    camera.start_preview()  # the start_preview() function
    while (1):
        a = 1

