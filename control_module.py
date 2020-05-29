

class ControlModule():
    def __init__(self,status='stop',camera_center_pid_output = 0,get_fream_interval = 1.5,ultrasonic_detect_interval = 0.05):
        # 小测运行状态
        self.status = status
        # 视频的宽和长
        w = 0
        h = 0

        #  用于控制运动的全局变量
        self.person_1 = []  # 人坐标
        self.forward_ultrasonic_distance = 0  # 前面超声波检测距离
        self.left_ultrasonic_distance = 0  # 左侧超声波检测距离

        # pid初始化参数
        self.kp = 0.09
        self.ki = 0.08
        self.kd = 0.002
        # 根据相机居中人物pid输出
        self.camera_center_pid_output = camera_center_pid_output

        # 超声波检测时间间隔
        self.ultrasonic_detect_interval =ultrasonic_detect_interval
        # 摄像头检测时间间隔(单位：秒)
        self.get_fream_interval = get_fream_interval