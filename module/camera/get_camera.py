"""
获取摄像头视频源
其他图像任务从此处获取源图像数据
"""
import cv2
import copy
import threading
import time
import os

class Video:

    # 使用单例模式
    # _instance = None
    # def __new__(cls, *args, **kw):
    #     if cls._instance is None:
    #         cls._instance = object.__new__(cls, *args, **kw)
    #     return cls._instance

    def __init__(self,camera_source):
        self.cap = cv2.VideoCapture(camera_source)  # camera_source为0，1..调用webcamera
        self.cap.set(cv2.CAP_PROP_FPS, 3)
        self.save_length=1000
        self.current_frame = None
        # 在线程中开启循环视频读取 其他取帧函数获取current_frame即可获取获取当前帧
        camera_loop_thread = threading.Thread(target=self.loop, args=(True,))
        camera_loop_thread.start()

    def save_image(self, frame):
        """
        保存图片 ，在目标文件夹超过指定数量时按时间顺序覆盖
        :param frame: 传入待写入图片
        :param save_struct_time: 图片保存结构化时间
        :return: None 写入图片
        """
        out_image_folder = 'images_out'
        sava_path = os.path.join(
            out_image_folder, time.strftime(
                "%Y%m%d%H%M%S", time.localtime()) + '.jpg')
        # print(sava_path)
        save_image_len = len(os.listdir(out_image_folder))
        if save_image_len >= self.save_length:
            images_list = [int(i.split('.')[0])
                           for i in os.listdir(out_image_folder)]
            images_list = sorted(images_list)
            os.remove(
                os.path.join(
                    out_image_folder, str(
                        images_list[0]) + '.jpg'))
        cv2.imwrite(sava_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

    def loop(self, show=False):
        while True:
            # print(type(self.current_frame))
            flag, frame = self.cap.read()
            self.current_frame = copy.deepcopy(frame)
            color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
            # print(ret, color_image.shape)
            if not flag:
                print('video over')
                break
            if show:
                cv2.imshow('frame',frame)
                cv2.waitKey(1)
        self.cap.release()
        cv2.destroyAllWindows()

    # 其他类调用该函数获取视频
    def get_frame(self):
        pass

camera_obj = Video(0)

if __name__ =='__main__':
    # 测试单例 需取消上面单例注释
    # a = Video()
    # b = Video()
    # print(id(a)==id(b))

    #
    c = Video(0)
    time.sleep(5)
    print(type(c.current_frame))


