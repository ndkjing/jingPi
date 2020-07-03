import cv2
#
# cap = cv2.VideoCapture(r'D:\dataset\crawler\row\videos\_(new).avi')
# i = 0
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# print(fps)
# cap.set(cv2.CAP_PROP_FPS,2)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# print(fps)
# while True:
#     flag,frame = cap.read()
#     if not flag:
#         print('video is over')
#         break
#     cv2.imshow('frame',frame)
#     cv2.waitKey(1)
# cap.release()
# cv2.destroyAllWindows()
# file_path=r'D:\dataset\crawler\row\images\actors\刘琳\ia_10026.jpg'
# # i =cv2.imread()
# # print(i)
# # import cv2
# import numpy as np
#
# cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
# cv2.imencode('.jpg', cv_img)[1].tofile('我.jpg')   # 写入中文路径






import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import time
import numpy as np
import face_alignment

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cpu')
image_path =r'C:\PythonProject\find_people\data\test_data\wjc.jpg'
input_img = io.imread(image_path)
start_time = time.time()
preds = fa.get_landmarks(input_img)[-1]
print('cost time ',time.time()-start_time)

# 2D-Plot
plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input_img)

for pred_type in pred_types.values():
    ax.plot(preds[pred_type.slice, 0],
            preds[pred_type.slice, 1],
            color=pred_type.color, **plot_style)

ax.axis('off')
# 3D-Plot
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax.scatter(preds[:, 0] * 1.2,
#                   preds[:, 1],
#                   preds[:, 2],
#                   c='cyan',
#                   alpha=1.0,
#                   edgecolor='b')
#
# for pred_type in pred_types.values():
#     ax.plot3D(preds[pred_type.slice, 0] * 1.2,
#               preds[pred_type.slice, 1],
#               preds[pred_type.slice, 2], color='blue')
#
# ax.view_init(elev=90., azim=90.)
# ax.set_xlim(ax.get_xlim()[::-1])
plt.show()


class FaceAlignerCv2:
    def __init__(self,keypoints, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=112, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.keypoints=np.asarray(keypoints)
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image,keypoints,ratio=None):
        # convert the landmark (x, y)-coordinates to a NumPy array

        # 68点对应坐标和5点对应坐标
        if (len(keypoints) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = 36, 42
            (rStart, rEnd) = 42, 48
        else:
            (lStart, lEnd) = 0, 1
            (rStart, rEnd) = 1, 2

        leftEyePts = keypoints[lStart:lEnd]
        rightEyePts = keypoints[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))
        print('angle',angle)
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        if not ratio:
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        else:
            (w, h) = (self.desiredFaceWidth*ratio, self.desiredFaceHeight*ratio)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

def enlarge_crop(bbox,ratio,image):
    """
    :param bbox:[x_min,y_min,x_max,y_max]
    :param ratio: 缩放比例
    :return: 缩放后的
    """
    h,w = image.shape[:2]
    print(h,w)
    enlarge_w = (bbox[2]-bbox[0])*(ratio-1)
    enlarge_h = (bbox[3]-bbox[1])*(ratio-1)
    enlarge_bbox = [bbox[0]-enlarge_w/2,bbox[1]-enlarge_h/2,bbox[2]+enlarge_w/2,bbox[3]+enlarge_h/2]
    print(enlarge_bbox)
    if enlarge_bbox[0]<0:
        enlarge_bbox[0]=0
    else:
        enlarge_bbox[0] = int(enlarge_bbox[0])

    if enlarge_bbox[1]<0:
        enlarge_bbox[1]=0
    else:
        enlarge_bbox[1] = int(enlarge_bbox[1])

    if enlarge_bbox[2] > w:
        enlarge_bbox[2] = int(w)
    else:
        enlarge_bbox[2] = int(enlarge_bbox[2])

    if enlarge_bbox[3] > h:
        enlarge_bbox[3] = int(h)
    else:
        enlarge_bbox[3] = int(enlarge_bbox[3])

    print(enlarge_bbox)
    return enlarge_bbox
ratio=3
obj = FaceAlignerCv2(preds)
image_cv2 = cv2.imread(image_path)
# print(min(preds[:,1]))
rect = [int(i) for i in [min(preds[:,0]),min(preds[:,1]),max(preds[:,0]),max(preds[:,1])]]
crop_image = image_cv2[rect[1]:rect[3],rect[0]:rect[2],:]
print(rect)
enlarge_bbox = enlarge_crop(rect,ratio,image_cv2)
cv2.imshow('crop_image',crop_image)
enlarge_crop_image = image_cv2[enlarge_bbox[1]:enlarge_bbox[3],enlarge_bbox[0]:enlarge_bbox[2],:]
cv2.imshow('enlarge_crop_image',enlarge_crop_image)
keypoints1 = obj.keypoints-np.asarray([int(min(preds[:,0])),int(min(preds[:,1]))])
out_image1 = obj.align(crop_image,keypoints1)
cv2.imshow('out_image1',out_image1)

keypoints2 = obj.keypoints-np.asarray([enlarge_bbox[0],enlarge_bbox[1]])
out_image2 = obj.align(enlarge_crop_image,keypoints2,ratio=ratio)
cv2.imshow('out_image2',out_image2)
print(out_image2.shape)
preds2 = fa.get_landmarks(out_image2[:,:,[2,0,1]])[-1]

rect2 = [int(i) for i in [min(preds2[:,0]),min(preds2[:,1]),max(preds2[:,0]),max(preds2[:,1])]]
if rect2[0] < 0:
    rect2[0] = 0
if rect2[1] < 0:
    rect2[1] = 0
if rect2[2] >out_image2.shape[1]:
    rect2[2] = int(out_image2.shape[1])
if rect2[3] > out_image2.shape[0]:
    rect2[3] = int(out_image2.shape[0])

crop_image3 = out_image2[rect2[1]:rect2[3],rect2[0]:rect2[2],:]
print('rect2',rect2)
# print(crop_image3)
cv2.imshow('out_image3',crop_image3)
cv2.waitKey(0)












