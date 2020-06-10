try:
    import tflite_runtime.interpreter as tflite
except:
    import tensorflow.lite as tflite
import platform
sysstr = platform.system()
if (sysstr == "Windows"):
    print("Call Windows tasks")
elif (sysstr == "Linux"):   # 树莓派也是Linux
    print("Call Linux tasks")
else:
    print("other System tasks")

import copy,os,sys
print('执行路径',os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))))
import time
import numpy as np
import cv2
from collections import deque

from module.camera.hand_keypoints.hand_tracker import HandTracker
from module.camera.hand_keypoints.gasture_utils.determine_gasture import create_known_finger_poses, determine_position, get_position_name_with_pose_id
from module.camera.hand_keypoints.gasture_utils.FingerPoseEstimate import FingerPoseEstimate

from module.camera.get_camera import camera_obj
"""
mediapipe 模型 handdetect模型
"""
class HandKeypoints:
    def __init__(self):
        self.palm_model_path = "./models/palm_detection_without_custom_op.tflite"
        self.landmark_model_path = "./models/mediapipe_hand_landmark.tflite"
        self.classify_model_path = "./models/mb2_classify.tflite"
        self.anchors_path = "./data/anchors.csv"
        self.save_length = 100  # 保存图片数量
        MIN_CONFIDENCE = 0.10
# cap = cv2.VideoCapture(r'C:\PythonProject\jing_vision\detection\keras_tf\tflite\posenet_mb2_ssd\dance.flv')
# cap = cv2.VideoCapture(r'hand.flv') # 使用本地视频
        self.cap = cv2.VideoCapture(0)  # 调用webcamera
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.box_enlarge=1.3
        self.detector = HandTracker(self.palm_model_path, self.landmark_model_path, self.anchors_path,
                               box_shift=0.2, box_enlarge=self.box_enlarge)
        self.Q = deque(maxlen=5)
        self.gasture_define = { 'Simple Thumbs Up':0,
                           'Thumbs Up Right':1,
                           'I love you':2,
                           'Victory':3  ,
                            'Pointing Up':4,
                            'Okay':5 ,
                           'Spock':6,
                           'One':7,
                           'Three':8
                           }
        self.reverse_gasture_define = {v:k for k,v in self.gasture_define.items()}
        self.known_finger_poses = create_known_finger_poses()
        #        8   12  16  20
        #        |   |   |   |
        #        7   11  15  19
        #    4   |   |   |   |
        #    |   6   10  14  18
        #    3   |   |   |   |
        #    |   5---9---13--17
        #    2    \         /
        #     \    \       /
        #      1    \     /
        #       \    \   /
        #        ------0-

        # Anatomy guide
        # http://blog.handcare.org/blog/2017/10/26/anatomy-101-finger-joints/
        self.HAND_POINTS = [
            "BASE",
            "T_STT", "T_BCMC", "T_MCP", "T_IP",  # Thumb
            "I_CMC", "I_MCP", "I_PIP", "I_DIP",  # Index
            "M_CMC", "M_MCP", "M_PIP", "M_DIP",  # Middle
            "R_CMC", "R_MCP", "R_PIP", "R_DIP",  # Ring
            "P_CMC", "P_MCP", "P_PIP", "P_DIP",  # Pinky
        ]
        # 关节连接
        self.limbs = [[0, 1],
                 [1, 2],
                 [2, 3],
                 [3, 4],
                 [0, 5],
                 [5, 6],
                 [6, 7],
                 [7, 8],
                 [0, 9],
                 [9, 10],
                 [10, 11],
                 [11, 12],
                 [0, 13],
                 [13, 14],
                 [14, 15],
                 [15, 16],
                 [0, 17],
                 [17, 18],
                 [18, 19],
                 [19, 20]
                 ]
        self.fps_list =[0 for i in range(10)]
        self.joint_color_code = [[139, 53, 255],
                            [0, 56, 255],
                            [43, 140, 237],
                            [37, 168, 36],
                            [147, 147, 0],
                            [70, 17, 145]]


        self.classify_interpreter = tflite.Interpreter(self.classify_model_path)
        self.classify_interpreter.allocate_tensors()

        # 获取输入和输出张量。
        self.classify_input_details = self.classify_interpreter.get_input_details()
        self.classify_output_details = self.classify_interpreter.get_output_details()
        print('classify_input_details',self.classify_input_details)
        print('classify_output_details',self.classify_output_details)

    def infer_video(self,save_iamge_flag=True):
        frame_count = 0
        return_track={}
        while True:
            ret, color_image=self.cap.read()
            show_image=copy.deepcopy(color_image)
            classify_image=copy.deepcopy(color_image)

            color_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)   # 将BGR转为RGB
            classify_image=cv2.cvtColor(classify_image, cv2.COLOR_BGR2RGB)   # 将BGR转为RGB
            # print(ret, color_image.shape)
            if not ret:
                print('video over')
                break
            start_time=time.time()
            # print(color_image.shape)
            # 判断检测间隔
            if frame_count%4==0:
                kp, box,return_track=self.detector(color_image,None,True)
                print('box:',box)
            elif return_track is None:
                pass
            else:
                kp, box, _ = self.detector(color_image,return_track,False,False)
            if kp is not None and box is not None:
                # print('kp is ',kp)
                pts = np.array(box, np.int32)
                kp = np.array(kp, np.int32)
                # 计算手势
                fingerPoseEstimate = FingerPoseEstimate(kp)
                fingerPoseEstimate.calculate_positions_of_fingers(print_finger_info=False)
                obtained_positions = determine_position(fingerPoseEstimate.finger_curled,
                                                        fingerPoseEstimate.finger_position, self.known_finger_poses,
                                                        0.45 * 10)
                # print(obtained_positions)
                # 根据字典的值进行排序
                gasture_pre = sorted(obtained_positions.items(), key=lambda item: item[1], reverse=True)
                # 仅绘制最高概率与绘制所有可能概率
                if len(gasture_pre) > 0:
                    self.Q.append(self.gasture_define[gasture_pre[0][0]])  # 通过队列中最多的元素输出为类别
                    counts = np.bincount(self.Q)
                    # 返回众数
                    max_index = np.argmax(counts)
                    print(self.reverse_gasture_define[max_index])
                    cv2.putText(show_image,
                                'rank 1 pre  %s probably %f' % (self.reverse_gasture_define[max_index], gasture_pre[0][1]),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1, cv2.LINE_AA)
                    # cv2.putText(show_image, 'rank 1  pre  %s   probably  %f'%(gasture_pre[0][0],gasture_pre[0][1]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1, cv2.LINE_AA)
                # for i,pre in enumerate(gasture_pre):
                #     cv2.putText(show_image, 'rank %d   pre：%s   probably%f'%(i+1,pre[0],pre[1]), (20, i*20+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1, cv2.LINE_AA)
                for i in kp:
                    cv2.circle(show_image, center=(i[0], i[1]), radius=3, color=(0, 255, 0), thickness=-1)
                # cv2.rectangle(show_image, (int(box[0,0]), int(box[0,1])), (int(box[1,0]), int(box[1,1])), (0, 255,255), 2)
                # 绘制关键点链接线
                cv2.line(show_image, (kp[0][0], kp[0][1]),
                         (kp[1][0], kp[1][1]), (255, 0, 0), 2)
                cv2.line(show_image, (kp[1][0], kp[1][1]),
                         (kp[2][0], kp[2][1]), (255, 0, 0), 2)
                cv2.line(show_image, (kp[2][0], kp[2][1]),
                         (kp[3][0], kp[3][1]), (255, 0, 0), 2)
                cv2.line(show_image, (kp[3][0], kp[3][1]),
                         (kp[4][0], kp[4][1]), (255, 0, 0), 2)

                cv2.line(show_image, (kp[0][0], kp[0][1]),
                         (kp[5][0], kp[5][1]), (255, 255, 0), 2)
                cv2.line(show_image, (kp[5][0], kp[5][1]),
                         (kp[6][0], kp[6][1]), (255, 255, 0), 2)
                cv2.line(show_image, (kp[6][0], kp[6][1]),
                         (kp[7][0], kp[7][1]), (255, 255, 0), 2)
                cv2.line(show_image, (kp[7][0], kp[7][1]),
                         (kp[8][0], kp[8][1]), (255, 255, 0), 2)

                cv2.line(show_image, (kp[5][0], kp[5][1]),
                         (kp[9][0], kp[9][1]), (0, 0, 255), 2)
                cv2.line(show_image, (kp[9][0], kp[9][1]),
                         (kp[10][0], kp[10][1]), (0, 0, 255), 2)
                cv2.line(show_image, (kp[10][0], kp[10][1]),
                         (kp[11][0], kp[11][1]), (0, 0, 255), 2)
                cv2.line(show_image, (kp[11][0], kp[11][1]),
                         (kp[12][0], kp[12][1]), (0, 0, 255), 2)

                cv2.line(show_image, (kp[9][0], kp[9][1]),
                         (kp[13][0], kp[13][1]), (255, 0, 255), 2)
                cv2.line(show_image, (kp[13][0], kp[13][1]),
                         (kp[14][0], kp[14][1]), (255, 0, 255), 2)
                cv2.line(show_image, (kp[14][0], kp[14][1]),
                         (kp[15][0], kp[15][1]), (255, 0, 255), 2)
                cv2.line(show_image, (kp[15][0], kp[15][1]),
                         (kp[16][0], kp[16][1]), (255, 0, 255), 2)

                cv2.line(show_image, (kp[0][0], kp[0][1]),
                         (kp[17][0], kp[17][1]), (0, 255, 0), 2)
                cv2.line(show_image, (kp[13][0], kp[13][1]),
                         (kp[17][0], kp[17][1]), (0, 255, 0), 2)
                cv2.line(show_image, (kp[17][0], kp[17][1]),
                         (kp[18][0], kp[18][1]), (0, 255, 0), 2)
                cv2.line(show_image, (kp[18][0], kp[18][1]),
                         (kp[19][0], kp[19][1]), (0, 255, 0), 2)
                cv2.line(show_image, (kp[19][0], kp[19][1]),
                         (kp[20][0], kp[20][1]), (0, 255, 0), 2)

                cv2.polylines(show_image, [pts], True, (0, 255, 255), 1)  # 绘制多边形
                # print(box)
                fps = 1 / (time.time() - start_time)
                print('FPS:', fps)
                index = frame_count % 10
                self.fps_list[index] = fps
                cv2.putText(show_image,
                            'FPS:%f' % (np.mean(self.fps_list)), (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 155), 1, cv2.LINE_AA)
            if save_iamge_flag:
                self.save_image(show_image)
            if sysstr == "Windows":
                cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("USB Camera", show_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def infer_frame(self,save_iamge_flag=False,show_flag = False):
        frame_count = 0
        return_track={}
        previous_frame,current_frame = None,None
        first_flag = True
        while True:
            current_frame = camera_obj.current_frame

            # 判断当前帧是否更新，未更新则等待更新后再预测
            # if first_flag:
            #     current_frame=camera_obj.current_frame
            # else:
            #     first_flag=False
            #     previous_frame = current_frame
            #     current_frame = camera_obj.current_frame
            # if (previous_frame==current_frame).all():
            #     time.sleep(1/10)
            #     continue
            show_image=copy.deepcopy(current_frame)
            classify_image=copy.deepcopy(current_frame)

            color_image=cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)   # 将BGR转为RGB
            classify_image=cv2.cvtColor(classify_image, cv2.COLOR_BGR2RGB)   # 将BGR转为RGB

            start_time=time.time()
            # print(color_image.shape)
            # 判断检测间隔
            kp, box,return_track=self.detector(color_image,None,True)
            print('box:',box)
            if return_track is None:
                pass
            else:
                kp, box, _ = self.detector(color_image,return_track,False,False)
            if kp is not None and box is not None:
                # print('kp is ',kp)
                pts = np.array(box, np.int32)
                kp = np.array(kp, np.int32)
                # 计算手势
                fingerPoseEstimate = FingerPoseEstimate(kp)
                fingerPoseEstimate.calculate_positions_of_fingers(print_finger_info=False)
                obtained_positions = determine_position(fingerPoseEstimate.finger_curled,
                                                        fingerPoseEstimate.finger_position, self.known_finger_poses,
                                                        0.45 * 10)
                # print(obtained_positions)
                # 根据字典的值进行排序
                gasture_pre = sorted(obtained_positions.items(), key=lambda item: item[1], reverse=True)
                # 仅绘制最高概率与绘制所有可能概率
                if len(gasture_pre) > 0:
                    self.Q.append(self.gasture_define[gasture_pre[0][0]])  # 通过队列中最多的元素输出为类别
                    counts = np.bincount(self.Q)
                    # 返回众数
                    max_index = np.argmax(counts)
                    print(self.reverse_gasture_define[max_index])
                    cv2.putText(show_image,
                                'rank 1 pre  %s probably %f' % (self.reverse_gasture_define[max_index], gasture_pre[0][1]),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1, cv2.LINE_AA)
                    # cv2.putText(show_image, 'rank 1  pre  %s   probably  %f'%(gasture_pre[0][0],gasture_pre[0][1]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1, cv2.LINE_AA)
                # for i,pre in enumerate(gasture_pre):
                #     cv2.putText(show_image, 'rank %d   pre：%s   probably%f'%(i+1,pre[0],pre[1]), (20, i*20+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1, cv2.LINE_AA)
                for i in kp:
                    cv2.circle(show_image, center=(i[0], i[1]), radius=3, color=(0, 255, 0), thickness=-1)
                # cv2.rectangle(show_image, (int(box[0,0]), int(box[0,1])), (int(box[1,0]), int(box[1,1])), (0, 255,255), 2)
                # 绘制关键点链接线
                cv2.line(show_image, (kp[0][0], kp[0][1]),
                         (kp[1][0], kp[1][1]), (255, 0, 0), 2)
                cv2.line(show_image, (kp[1][0], kp[1][1]),
                         (kp[2][0], kp[2][1]), (255, 0, 0), 2)
                cv2.line(show_image, (kp[2][0], kp[2][1]),
                         (kp[3][0], kp[3][1]), (255, 0, 0), 2)
                cv2.line(show_image, (kp[3][0], kp[3][1]),
                         (kp[4][0], kp[4][1]), (255, 0, 0), 2)

                cv2.line(show_image, (kp[0][0], kp[0][1]),
                         (kp[5][0], kp[5][1]), (255, 255, 0), 2)
                cv2.line(show_image, (kp[5][0], kp[5][1]),
                         (kp[6][0], kp[6][1]), (255, 255, 0), 2)
                cv2.line(show_image, (kp[6][0], kp[6][1]),
                         (kp[7][0], kp[7][1]), (255, 255, 0), 2)
                cv2.line(show_image, (kp[7][0], kp[7][1]),
                         (kp[8][0], kp[8][1]), (255, 255, 0), 2)

                cv2.line(show_image, (kp[5][0], kp[5][1]),
                         (kp[9][0], kp[9][1]), (0, 0, 255), 2)
                cv2.line(show_image, (kp[9][0], kp[9][1]),
                         (kp[10][0], kp[10][1]), (0, 0, 255), 2)
                cv2.line(show_image, (kp[10][0], kp[10][1]),
                         (kp[11][0], kp[11][1]), (0, 0, 255), 2)
                cv2.line(show_image, (kp[11][0], kp[11][1]),
                         (kp[12][0], kp[12][1]), (0, 0, 255), 2)

                cv2.line(show_image, (kp[9][0], kp[9][1]),
                         (kp[13][0], kp[13][1]), (255, 0, 255), 2)
                cv2.line(show_image, (kp[13][0], kp[13][1]),
                         (kp[14][0], kp[14][1]), (255, 0, 255), 2)
                cv2.line(show_image, (kp[14][0], kp[14][1]),
                         (kp[15][0], kp[15][1]), (255, 0, 255), 2)
                cv2.line(show_image, (kp[15][0], kp[15][1]),
                         (kp[16][0], kp[16][1]), (255, 0, 255), 2)

                cv2.line(show_image, (kp[0][0], kp[0][1]),
                         (kp[17][0], kp[17][1]), (0, 255, 0), 2)
                cv2.line(show_image, (kp[13][0], kp[13][1]),
                         (kp[17][0], kp[17][1]), (0, 255, 0), 2)
                cv2.line(show_image, (kp[17][0], kp[17][1]),
                         (kp[18][0], kp[18][1]), (0, 255, 0), 2)
                cv2.line(show_image, (kp[18][0], kp[18][1]),
                         (kp[19][0], kp[19][1]), (0, 255, 0), 2)
                cv2.line(show_image, (kp[19][0], kp[19][1]),
                         (kp[20][0], kp[20][1]), (0, 255, 0), 2)

                cv2.polylines(show_image, [pts], True, (0, 255, 255), 1)  # 绘制多边形
                # print(box)
                fps = 1 / (time.time() - start_time)
                print('FPS:', fps)
                index = frame_count % 10
                self.fps_list[index] = fps
                cv2.putText(show_image,
                            'FPS:%f' % (np.mean(self.fps_list)), (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 155), 1, cv2.LINE_AA)
            if save_iamge_flag:
                self.save_image(show_image)
            if show_flag:
                cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("USB Camera", show_image)
                cv2.waitKey(1)



        # 保存图片

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


if __name__ =='__main__':
    test_obj = HandKeypoints()
    test_obj.infer_frame(show_flag=True)
