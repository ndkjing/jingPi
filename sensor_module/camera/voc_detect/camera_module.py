import numpy as np
import argparse
import cv2
import os.path
import sys



# 暂时报错
# OPENCV_OBJECT_TRACKERS = {
#     "csrt": cv2.TrackerCSRT_create,
#     "kcf": cv2.TrackerKCF_create,
#     "boosting": cv2.TrackerBoosting_create,
#     "mil": cv2.TrackerMIL_create,
#     "tld": cv2.TrackerTLD_create,
#     "medianflow": cv2.TrackerMedianFlow_create,
#     "mosse": cv2.TrackerMOSSE_create
# }


class CameraModule():
    # 初始化传感器
    def __init__(self):
        # 配置信息
        self.prototxt_path = r'module/voc_detect/mobilenet_ssd/MobileNetSSD_deploy.prototxt'
        self.model_path = r'module/voc_detect/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
        self.confidence_threshold = 0.2
        # VOC 类别
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
        self.init_bb = None   #初始化框
        # TODO  使用tracker
        #self.tracker = OPENCV_OBJECT_TRACKERS['csrt']()  # 追踪器


    def get_person_center(self, image):  # 传入视频帧检测目标
        (image_h, image_w) = image.shape[:2]   #获取长和宽
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        print("[INFO] computing object detections...")
        self.net.setInput(blob)
        detections = self.net.forward()
        # loop over the detections
        return_list=[]
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([image_w, image_h, image_w, image_h])
                (startX, startY, endX, endY) = box.astype("int")
                if self.CLASSES[idx] == 'person':   #如果检测到人，根据输入图片缩放x，y然后返回人矩形框坐标
                    """
                    直接在box出缩放回原图不用再次执行缩放
                    width_scale = w/300   # 
                    hight_scale = h/300
                    
                    center_x =  width_scale*(startX+endX)/2
                    center_y = hight_scale*(startY+endY)/2
                    bbox_w = width_scale*(endX-startX)
                    bbox_h =hight_scale*(endY-startY)
                    """
                    self.init_bb = (startX, startY, endX - startX, endY - startY)  # 初始化init_bb追踪框  追踪不适用图像缩放 此处传入原图bbox
                    return_list.append([(startX+endX)/2,(startY+endY)/2,(endX-startX),(endY-startY)])
                    print('person')
        return return_list

    # def track_person(self,image):  # 传入视频帧追踪目标
    #     (image_h, image_w) = image.shape[:2]
    #     return_list = []
    #     if self.init_bb is not None:
    #         self.tracker.init(image, self.init_bb)
    #         # grab the new bounding box coordinates of the object
    #         (success, box) = self.tracker.update(image)
    #
    #         # check to see if the tracking was a success
    #         if success:
    #             (x, y, w, h) = [int(v) for v in box]   # print('track:',(x, y), (x + w, y + h))
    #
    #             width_scale = image_w / 300
    #             hight_scale =image_h / 300
    #             center_x = width_scale * (x + w/2)
    #             center_y = hight_scale * (y + h/2)
    #             return_list.append([center_x,center_y,w,h])
    #             print('person')
    #     return return_list




