# 导入工具包
from collections import OrderedDict
import numpy as np
import dlib
import cv2

# 有序字典
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


class picture:

    #得到各点的坐标
    def shape_to_np(shape, dtype="int"):
        # 创建68*2
        #返回一个给定形状和类型初始化用0填充的数组
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        # 遍历每一个关键点
        # 得到x与y坐标
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    #得到整张脸的五官区域，得到最后叠加的图片
    def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
        # 创建两个copy
        overlay = image.copy()
        output = image.copy()
        # 设置一些颜色区域
        if colors is None:
            colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                      (168, 100, 168), (158, 163, 32),
                      (163, 38, 32), (180, 42, 220)]
        # 遍历每一个五官区域
        for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
            # 得到每一个点的坐标
            (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
            pts = shape[j:k]
            # 两种情况，一个下颚，一个是其他
            if name == "jaw":  #如果是下巴局部
                # 用线条连起来
                for l in range(1, len(pts)):
                    ptA = tuple(pts[l - 1])
                    ptB = tuple(pts[l])
                    #设置划线的的参数，点，颜色和粗细
                    cv2.line(overlay, ptA, ptB, colors[i], 2)
            # 计算凸包
            else:  #如果整个脸型
                hull = cv2.convexHull(pts)
                cv2.drawContours(overlay, [hull], -1, colors[i], -1)
        # 将画了的图片与原图叠加显示，alpha为比例
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        #cv2.imshow(overlay)
        return output

    if __name__ == "__main__":
        # 加载人脸检测与关键点定位
        detector = dlib.get_frontal_face_detector()
        #获得脸部关键点位置检测器
        predictor = dlib.shape_predictor('D:/ComputerView/data/shape_predictor_68_face_landmarks.dat')

        # 读取输入数据，预处理
        image = cv2.imread('D:/Python/image/zhangWang.jpg')
        #获得彩色图片的高和宽通道
        (h, w) = image.shape[:2]
        width = 500
        r = width / float(w)
        dim = (width, int(h * r))  #得到的新坐标
        print('00000000000000000')
        print(dim)
        #将图片按指定格式显示，使用区域插植算法，以达到减少图像失真和噪声的目的
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #将彩色图片转化为灰度图片

        # 人脸检测
        rects = detector(gray, 1)

        # 遍历检测到的框
        for (i, rect) in enumerate(rects):
            # 对人脸框进行关键点定位
            # 转换成坐标数组ndarray
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            # 遍历每一个部分
            for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
                clone = image.copy()
                #在指定的图片及位置，显示相应的名字，指定字体，大小，颜色，粗细
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0,0), 2)

                # 根据位置画点，设置点的位置，大小，颜色等
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 2, (0, 0, 255), -1)

                # 用一个最小的举行，把找到的区域包起来
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                #显示相应的区域
                roi = image[y:y + h, x:x + w]
                # #获得彩色图片的宽和高的通道
                (h, w) = roi.shape[:2]
                width = 250
                r = width / float(w)
                dim = (width, int(h * r))
                #显示得到的区域，使用区域插植算法，以达到减少图像失真和噪声的目的
                roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)

                # 显示每一部分
                cv2.imshow("ROI", roi)
                cv2.imshow("Image", clone)
                cv2.waitKey(0)

            # 展示所有区域
            output = visualize_facial_landmarks(image, shape)
            cv2.imshow("Image", output)
            cv2.waitKey(0)
