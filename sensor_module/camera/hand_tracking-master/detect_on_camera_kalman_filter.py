import platform
sysstr = platform.system()
if (sysstr == "Windows"):
    print("Call Windows tasks")
elif (sysstr == "Linux"):   # 树莓派也是Linux
    print("Call Linux tasks")
else:
    print("other System tasks")

import math,re,copy
import time
import numpy as np
import cv2
from collections import deque

from hand_tracker import HandTracker

from gasture_utils.determine_gasture import create_known_finger_poses, determine_position, get_position_name_with_pose_id
from gasture_utils.FingerPoseEstimate import FingerPoseEstimate
"""
mediapipe 模型 handdetect模型
 与直接检测相比多了使用卡尔曼滤波预测关键节点链接线
"""
palm_model_path = "./models/palm_detection_without_custom_op.tflite"
landmark_model_path = "./models/mediapipe_hand_landmark.tflite"
anchors_path = "./data/anchors.csv"
MIN_CONFIDENCE = 0.10
# cap = cv2.VideoCapture(r'C:\PythonProject\jing_vision\detection\keras_tf\tflite\posenet_mb2_ssd\dance.flv')
# cap = cv2.VideoCapture(r'hand.flv') # 使用本地视频
cap = cv2.VideoCapture(0)  # 调用webcamera
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                       box_shift=0.2, box_enlarge=1.3)

Q = deque(maxlen=5)  # 用有限队列求众数来防止噪音
gasture_define = { 'Simple Thumbs Up':0,
                   'Thumbs Up Right':1,
                   'I love you':2,
                   'Victory':3  ,
                    'Pointing Up':4,
                    'Okay':5 ,
                   'Spock':6,
                   }
reverse_gasture_define = {v:k for k,v in gasture_define.items()}
known_finger_poses = create_known_finger_poses()
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

# Anatomy guide http://blog.handcare.org/blog/2017/10/26/anatomy-101-finger-joints/
HAND_POINTS = [
    "BASE",
    "T_STT", "T_BCMC", "T_MCP", "T_IP",  # Thumb
    "I_CMC", "I_MCP", "I_PIP", "I_DIP",  # Index
    "M_CMC", "M_MCP", "M_PIP", "M_DIP",  # Middle
    "R_CMC", "R_MCP", "R_PIP", "R_DIP",  # Ring
    "P_CMC", "P_MCP", "P_PIP", "P_DIP",  # Pinky
]
limbs = [[0, 1],
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

fps_list =[0 for i in range(10)]
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]
frame_count = 0
return_track={}
# 卡尔曼滤波
kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(21)]
for _, joint_kalman_filter in enumerate(kalman_filter_array):
    joint_kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                    np.float32)
    joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                   np.float32) * 3e-2

kp, box,return_track = None,None,None
while True:
    ret, color_image = cap.read()
    show_image = copy.deepcopy(color_image)
    color_image = cv2.cvtColor(color_image , cv2.COLOR_BGR2RGB)   # 将BGR转为RGB
    if not ret:
        print('video over')
        break
    start_time = time.time()
    # 判断检测间隔
    if frame_count%3==0:
        print('detect hand')
        kp, box,return_track=detector(color_image,None,True)
    elif return_track is None:   # 未检测到手
        pass
    else:
        pass
        # kp, box, _ = detector(color_image,return_track,False)

    if kp is not None and box is not None:
        # print('kp is ',kp)
        pts=np.array(box, np.int32)
        kp=np.array(kp, np.int32)
        joint_coord_set = np.zeros((21, 2))
        for index, coord in enumerate(kp):
            joint_coord = np.asarray([[coord[0]], [coord[1]]], dtype=np.float32)
            kalman_filter_array[index].correct(joint_coord)
            kalman_pred = kalman_filter_array[index].predict()
            joint_coord_set[index, :] = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
            # kp[index] = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
            cv2.circle(show_image, center=(kalman_pred[0], kalman_pred[1]), radius=3, color=(255, 0, 0), thickness=-1)  # 显示关节节点

        for limb_num in range(len(limbs)):
            x1 = joint_coord_set[limbs[limb_num][0], 0]
            y1 = joint_coord_set[limbs[limb_num][0], 1]
            x2 = joint_coord_set[limbs[limb_num][1], 0]
            y2 = joint_coord_set[limbs[limb_num][1], 1]
            # 将预测值在未检测时赋值给检测值

            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            cv2.line(show_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.polylines(show_image, [pts], True, (0, 0, 255), 1)  # 绘制多边形
        # 计算手势
        fingerPoseEstimate = FingerPoseEstimate(kp)
        fingerPoseEstimate.calculate_positions_of_fingers(print_finger_info=False)
        obtained_positions = determine_position(fingerPoseEstimate.finger_curled,
                                                fingerPoseEstimate.finger_position, known_finger_poses,
                                                0.45 * 10)
        # print(obtained_positions)
        # 根据字典的值进行排序
        gasture_pre = sorted(obtained_positions.items(), key=lambda item: item[1], reverse=True)
        # 仅绘制最高概率与绘制所有可能概率
        if len(gasture_pre)>0:
            Q.append(gasture_define[gasture_pre[0][0]])  # 通过队列中最多的元素输出为类别
            counts = np.bincount(Q)
            # 返回众数
            max_index = np.argmax(counts)
            print(reverse_gasture_define[max_index])
            cv2.putText(show_image, 'rank 1 pre  %s probably %f'%(reverse_gasture_define[max_index],gasture_pre[0][1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1, cv2.LINE_AA)

    # 显示平均10次检测fps
    fps = min(1 / (time.time() - start_time+1e-10),30)
    print('FPS:', fps)
    index = frame_count % 10
    fps_list[index] = fps
    cv2.putText(show_image,
                'FPS:%f' % (np.mean(fps_list)), (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 155), 1, cv2.LINE_AA)

    frame_count += 1
    cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("USB Camera", show_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()