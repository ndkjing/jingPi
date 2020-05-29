try:
    import tflite_runtime.interpreter as tflite
except:
    import tensorflow as tf
import platform
sysstr = platform.system()
if (sysstr == "Windows"):
    print("Call Windows tasks")
elif (sysstr == "Linux"):   # 树莓派也是Linux
    print("Call Linux tasks")
else:
    print("other System tasks")

import copy,os
import time
import numpy as np
import cv2
from collections import deque

from hand_tracker import HandTracker

from gasture_utils.determine_gasture import create_known_finger_poses, determine_position, get_position_name_with_pose_id
from gasture_utils.FingerPoseEstimate import FingerPoseEstimate

"""
mediapipe 模型 handdetect模型
"""

palm_model_path = "./models/palm_detection_without_custom_op.tflite"
landmark_model_path = "./models/mediapipe_hand_landmark.tflite"
classify_model_path = "./models/mb2_classify.tflite"
anchors_path = "./data/anchors.csv"
MIN_CONFIDENCE = 0.10
# cap = cv2.VideoCapture(r'C:\PythonProject\jing_vision\detection\keras_tf\tflite\posenet_mb2_ssd\dance.flv')
# cap = cv2.VideoCapture(r'hand.flv') # 使用本地视频
cap = cv2.VideoCapture(0)  # 调用webcamera
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
box_enlarge=1.3
detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                       box_shift=0.2, box_enlarge=box_enlarge)


Q = deque(maxlen=5)
gasture_define = { 'Simple Thumbs Up':0,
                   'Thumbs Up Right':1,
                   'I love you':2,
                   'Victory':3  ,
                    'Pointing Up':4,
                    'Okay':5 ,
                   'Spock':6,
                   'One':7,
                   'Three':8
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

# Anatomy guide
# http://blog.handcare.org/blog/2017/10/26/anatomy-101-finger-joints/
HAND_POINTS = [
    "BASE",
    "T_STT", "T_BCMC", "T_MCP", "T_IP",  # Thumb
    "I_CMC", "I_MCP", "I_PIP", "I_DIP",  # Index
    "M_CMC", "M_MCP", "M_PIP", "M_DIP",  # Middle
    "R_CMC", "R_MCP", "R_PIP", "R_DIP",  # Ring
    "P_CMC", "P_MCP", "P_PIP", "P_DIP",  # Pinky
]
# 关节连接
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

# TODO
classify_interpreter = tflite.Interpreter(classify_model_path)
classify_interpreter.allocate_tensors()

# 获取输入和输出张量。
classify_input_details = classify_interpreter.get_input_details()
classify_output_details = classify_interpreter.get_output_details()
print('classify_input_details',classify_input_details)
print('classify_output_details',classify_output_details)

frame_count = 0
return_track={}
while True:
    ret, color_image=cap.read()
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
    if frame_count%2==0:
        kp, box,return_track=detector(color_image,None,True)
    elif return_track is None:
        pass
    else:
        kp, box, _ = detector(color_image,return_track,False,False)
    if box is not None:
        # print('kp is ',kp)
        pts=np.array(box, np.int32)
        # print(pts)
        # cv2.rectangle(show_image, (int(box[0,0]), int(box[0,1])), (int(box[1,0]), int(box[1,1])), (0, 255,255), 2)
        x_min = max(0,np.min(pts[:,0]))
        x_max = np.max(pts[:,0])
        y_min = max(0,np.min(pts[:,1]))
        y_max = np.max(pts[:,1])
        center_x,center_y = int((x_min+x_max)/2),int((y_min+y_max)/2)
        box_width,box_height = (x_max-x_min),(y_max-y_min)
        shrink_x_min, shrink_x_max, shrink_y_min, shrink_y_max = center_x-int((box_width/2)/box_enlarge),center_x+int((box_width/2)/box_enlarge),center_y-int((box_height/2)/box_enlarge),center_y+int((box_height/2)/box_enlarge)
        # print(x_min,x_max,y_min,y_max)
        # classify_image = cv2.resize(classify_image[shrink_y_min:shrink_y_max,shrink_x_min:shrink_x_max,:],(224,224))
        classify_image = cv2.resize(classify_image[y_min:y_max,x_min:x_max,:],(224,224))
        classify_image = np.expand_dims(np.array(classify_image/255.0,dtype=np.float32),axis=0)
        classify_interpreter.set_tensor(classify_input_details[0]['index'], classify_image)
        classify_interpreter.invoke()
        tflite_results = classify_interpreter.get_tensor(classify_output_details[0]['index'])
        print(np.argmax(tflite_results)+1)
        # cv2.polylines(show_image, [pts], True, (0, 255, 255), 1)  # 绘制多边形
        cv2.rectangle(show_image,(x_min,y_min),(x_max,y_max),(0,0,255),3)
        cv2.rectangle(show_image,(shrink_x_min,shrink_y_min),(shrink_x_max,shrink_y_max),(0,0,255),3)
        # print(box)
        fps = 1 / (time.time() - start_time)
        # print('FPS:', fps)
        index = frame_count%10
        fps_list[index]=fps
        cv2.putText(show_image,
                    'FPS:%f' % ( np.mean(fps_list)), (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 155), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join('./images/%d.jpg'%frame_count),show_image)
    if sysstr == "Windows":
        cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("USB Camera", show_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_count +=1

cap.release()
cv2.destroyAllWindows()
