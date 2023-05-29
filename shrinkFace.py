# -*- coding:gb18030 -*-

import dlib
import cv2
import numpy as np
import math

predictor_path = 'D:/ComputerView/data/shape_predictor_68_face_landmarks.dat'

# ʹ��dlib�Դ���frontal_face_detector��Ϊ���ǵ�������ȡ��
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        # for idx,point in enumerate(land_marks_node):
        #     # 68������
        #     pos = (point[0,0],point[0,1])
        #     print(idx,pos)
        #     # ����cv2.circle��ÿ�������㻭һ��Ȧ����68��
        #     cv2.circle(img_src, pos, 5, color=(0, 255, 0))
        #     # ����cv2.putText���1-68
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(img_src, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        land_marks.append(land_marks_node)

    return land_marks


'''
������ Interactive Image Warping �ֲ�ƽ���㷨
'''


def localTranslationWarp(srcImg, startX, startY, endX, endY, radius):
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    # ���㹫ʽ�е�|m-c|^2
    ddmc = (endX-50 - startX) * (endX-50 - startX) + (endY-50 - startY) * (endY-50 - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            # ����õ��Ƿ����α�Բ�ķ�Χ֮��
            # �Ż�����һ����ֱ���ж��ǻ��ڣ�startX,startY)�ľ������
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue

            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

            if (distance < ddradius):
                # �������i,j�������ԭ����
                # ���㹫ʽ���ұ�ƽ������Ĳ���
                ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                ratio = ratio * ratio

                # ӳ��ԭλ��
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)

                # ����˫���Բ�ֵ���õ�UX��UY��ֵ
                value = BilinearInsert(srcImg, UX, UY)
                # �ı䵱ǰ i ��j��ֵ
                copyImg[j, i] = value

    return copyImg


# ˫���Բ�ֵ��
def BilinearInsert(src, ux, uy):
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1].astype(np.float64) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float64) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float64) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float64) * (ux - float(x1)) * (uy - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)


def face_thin_auto(src):
    landmarks = landmark_dec_dlib_fun(src)

    # ���δ��⵽�����ؼ��㣬�Ͳ���������
    if len(landmarks) == 0:
        return

    for landmarks_node in landmarks:
        left_landmark = landmarks_node[3]
        left_landmark_down = landmarks_node[5]

        right_landmark = landmarks_node[13]
        right_landmark_down = landmarks_node[15]

        endPt = landmarks_node[30]

        # �����4���㵽��6����ľ�����Ϊ��������
        r_left = math.sqrt(
            (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
            (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))

        # �����14���㵽��16����ľ�����Ϊ��������
        r_right = math.sqrt(
            (right_landmark[0, 0] - right_landmark_down[0, 0]) * (right_landmark[0, 0] - right_landmark_down[0, 0]) +
            (right_landmark[0, 1] - right_landmark_down[0, 1]) * (right_landmark[0, 1] - right_landmark_down[0, 1]))

        # �������
        thin_image = localTranslationWarp(src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1],
                                          r_left)
        # ���ұ���
        # thin_image = localTranslationWarp(thin_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0],
        #                                   endPt[0, 1], r_right)

    # ��ʾ
    cv2.imshow('thin', thin_image)
    cv2.imwrite('thin.jpg', thin_image)


def main():
    src = cv2.imread('D:/ComputerView/facial1.jpg')
    cv2.imshow('src', src)
    face_thin_auto(src)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
