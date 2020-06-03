# from module.camera.voc_detect
from module.camera.hand_keypoints import infer

"""
hand keypoints 
"""
hand_detect = infer.DetectHand()
def detect_img(img):
    out = hand_detect.run()
    return out




