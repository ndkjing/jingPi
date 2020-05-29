import RPi.GPIO as GPIO
import time


class WheelModule:
    # 初始化
    def __init__(self, PIN_IN1_L, PIN_IN2_L, PIN_IN1_R, PIN_IN2_R):
        print('Wheel Module In Progress')
        GPIO.setmode(GPIO.BOARD)
        self.PIN_IN1_L = PIN_IN1_L
        self.PIN_IN2_L = PIN_IN2_L
        self.PIN_IN1_R = PIN_IN1_R
        self.PIN_IN2_R = PIN_IN2_R
        self.setup()

    def setup(self):
        GPIO.setup(self.PIN_IN1_L, GPIO.OUT)
        GPIO.setup(self.PIN_IN2_L, GPIO.OUT)
        GPIO.setup(self.PIN_IN1_R, GPIO.OUT)
        GPIO.setup(self.PIN_IN2_R, GPIO.OUT)


    # 前进的代码
    def forward(self):
        GPIO.output(self.PIN_IN1_L, GPIO.LOW)
        GPIO.output(self.PIN_IN2_L, GPIO.HIGH)
        GPIO.output(self.PIN_IN1_R, GPIO.LOW)
        GPIO.output(self.PIN_IN2_R, GPIO.HIGH)


    # 后退
    def backward(self):
        GPIO.output(self.PIN_IN1_L, GPIO.HIGH)
        GPIO.output(self.PIN_IN2_L, GPIO.LOW)
        GPIO.output(self.PIN_IN1_R, GPIO.HIGH)
        GPIO.output(self.PIN_IN2_R, GPIO.LOW)


    # 左转
    def left(self):
        GPIO.output(self.PIN_IN1_L, GPIO.LOW)
        GPIO.output(self.PIN_IN2_L, GPIO.LOW)
        GPIO.output(self.PIN_IN1_R, GPIO.LOW)
        GPIO.output(self.PIN_IN2_R, GPIO.HIGH)


    # 右转
    def right(self):
        GPIO.output(self.PIN_IN1_L, GPIO.LOW)
        GPIO.output(self.PIN_IN2_L, GPIO.HIGH)
        GPIO.output(self.PIN_IN1_R, GPIO.LOW)
        GPIO.output(self.PIN_IN2_R, GPIO.LOW)


    # 停止
    def stop(self):
        GPIO.output(self.PIN_IN1_L, GPIO.LOW)
        GPIO.output(self.PIN_IN2_L, GPIO.LOW)
        GPIO.output(self.PIN_IN1_R, GPIO.LOW)
        GPIO.output(self.PIN_IN2_R, GPIO.LOW)

    def quit(self):
        GPIO.cleanup()

if __name__ == "__main__":
    try:

        m = WheelModule(11, 12, 13, 15)

        print('forward')
        m.forward()
        time.sleep(0.3)
        # m.stop()

        print('backward')
        m.backward()
        time.sleep(0.3)

        print('left')
        m.left()
        time.sleep(0.2)

        print('right')
        m.right()
        time.sleep(0.3)

        print('backward')
        m.backward()
        time.sleep(0.3)



    except KeyboardInterrupt:
        pass
    GPIO.cleanup()