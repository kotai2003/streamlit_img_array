import cv2
import numpy as np
import math


class edge():

    def __init__(self,kernel1=5,thres1_val=1, thres2_val=1, kernel2=1, iter_dialate=1, iter_erode=1):

       self.kernel1 = kernel1
       self.thres1_val = thres1_val
       self.thres2_val = thres2_val
       self.kernel2 = kernel2
       self.iter_dialate = iter_dialate
       self.iter_erode = iter_erode


    def preprocess(self, img_array, rot_state=False, trans_state = False):
        #img is considered to be (B,G,R), which is opened with OpenCV.

        #0. if the image is gray scale image, the image will be converted to color image.

        if len(img_array.shape) < 3 :
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)


        height, width, channel = img_array.shape
        # img_size = height * width

        # canny
        # 00.make a copy
        img_copy = img_array.copy()  # for contour
        img_copy2 = img_array.copy()  # for transforamtaion
        # img_black = np.zeros_like(img_array)

        # 01. Color -> Gray Scale
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        # 02. Gaussian Blur
        kernelb = (self.kernel1 * 2) + 1
        img_blur = cv2.GaussianBlur(img_gray, (kernelb, kernelb), None)

        # 03 Canny Edge Detection
        img_edge = cv2.Canny(img_blur, threshold1=self.thres1_val, threshold2=self.thres2_val)

        # 04. dilate , erode
        kernel2 = (self.kernel2 * 2) + 1
        k_mat = np.ones((kernel2, kernel2), np.uint8)

        # 04.01. dilate
        img_dialate = cv2.dilate(img_edge, kernel=k_mat, iterations=self.iter_dialate)

        # 04.02. Erode
        img_erode = cv2.erode(img_dialate, kernel=k_mat, iterations=self.iter_erode)

        # 04.03. Binary Threshold + Otsu Threshold
        _, img_thresh = cv2.threshold(img_erode, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # img_cont = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), -1)
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        #--------------------------------------------------
        # Mask Pattern
        # --------------------------------------------------
        mask = np.zeros_like(img_gray)
        img_mask = cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)

        # 6.1. Moment -> calc rotate angle and center of gravity
        m = cv2.moments(img_mask)
        # m_area = m['m00']
        x_g = m['m10'] / m['m00']
        y_g = m['m01'] / m['m00']

        ang_deg = 0.5 * math.degrees(math.atan2(2.0 * m['mu11'], m['mu20'] - m['mu02']))

        # 6.2. Rotation of mask, rotation of original_image
        center = (x_g, y_g)
        rot_size = (img_array.shape[1], img_array.shape[0])

        if rot_state == True:
            rot_mat = cv2.getRotationMatrix2D(center=center, angle=ang_deg, scale=1)
        else:
            rot_mat = cv2.getRotationMatrix2D(center=center, angle=0, scale=1)

        # img_rot: color photo, img_rot_mask: img_mask
        img_rot = cv2.warpAffine(img_copy2, rot_mat, rot_size, flags=cv2.INTER_CUBIC)
        img_rot_mask = cv2.warpAffine(img_mask, rot_mat, rot_size, flags=cv2.INTER_CUBIC)

        # 6.3. Transportation
        if trans_state == True:
            delta_x = int(0.5 * width - x_g)
            delta_y = int(0.5 * height - y_g)
        else:
            delta_x = 0
            delta_y = 0

        trans_mat = np.float32([[1, 0, delta_x], [0, 1, delta_y]])

        img_trans = cv2.warpAffine(img_rot, trans_mat, (width, height))
        img_trans_mask = cv2.warpAffine(img_rot_mask, trans_mat, (width, height))

        # 6.4. ちょっと整理
        img_tf = img_trans  # color　image , tranformation finished
        img_tf_mask = img_trans_mask  # gray scale

        # mask操作
        # 6.1.　(linear transformed original) AND (lr_tf_mask)
        # 6.2.  np.ones（背景) AND not(lr_tf_mask)
        # 6.3.  6.1 OR 6.2

        # 6.1. BitWise And
        img_61 = cv2.bitwise_and(img_tf, cv2.cvtColor(img_tf_mask, cv2.COLOR_GRAY2BGR))

        # 6.2. BitWise AND
        img_62 = cv2.bitwise_and((255 * np.ones_like(img_gray)), (cv2.bitwise_not(img_tf_mask)))
        img_62 = cv2.cvtColor(img_62, cv2.COLOR_GRAY2BGR)

        img_AND = cv2.bitwise_and(img_trans, cv2.cvtColor(img_trans_mask, cv2.COLOR_GRAY2BGR))

        # 6.3. OR
        img_63 = cv2.bitwise_or(img_61, img_62)  # Final

        del img_array, img_copy, img_mask, img_blur , img_copy2, img_AND, img_62, img_dialate, \
            img_edge, img_erode, img_gray, img_rot, img_rot_mask, img_tf, img_tf_mask, \
            img_trans, img_trans_mask

        #BGR 2 RGB
        return cv2.cvtColor(img_61,cv2.COLOR_BGR2RGB ), cv2.cvtColor( img_63 ,cv2.COLOR_BGR2RGB)








