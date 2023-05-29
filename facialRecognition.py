# coding:utf-8
import cv2
import dlib
def read():  #从摄像头中获取照片
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    flag = cap.isOpened()
    index = 1
    while (flag):
        ret, frame = cap.read()
        cv2.imshow("Capture_Paizhao", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):  # 按下s键，进入下面的保存图片操作
            cv2.imwrite("D:/ComputerView/facial/face" + str(index) + ".jpg", frame)
            print("save" + str(index) + ".jpg successfuly!")
            print("-------------------------")
            index += 1
        elif k == ord('q'):  # 按下q键，程序退出
            break
    cap.release() # 释放摄像头
    cv2.destroyAllWindows()# 释放并销毁窗口


def test():
    img = cv2.imread('D:/ComputerView/facial/face1.jpg')  # 读取图片

    detector = dlib.get_frontal_face_detector()  # 加载人脸检测器

    dets = detector(img, 1)  # 获取人脸的区域，保存在dets中

    print("检测到的人脸数目: {}".format(len(dets)))  # 输出人脸区域的信息
    for i, d in enumerate(dets):
        print("所在区域： {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))

if __name__ == '__main__':
    read()
    test()

