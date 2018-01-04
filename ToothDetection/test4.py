import cv2
import numpy as np
import glob


class MyObj:
    template = None
    w = None
    h = None

img_rgb = cv2.imread('Teeth_Target/Target.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

files = glob.glob("./Tooth_images/*")
MyTemplateList = []
for pth in files:
    temp = MyObj()
    temp.template = cv2.imread(pth, 0)
    w, h = temp.template.shape[::-1]
    temp.w = w
    temp.h = h
    MyTemplateList.append(temp)

print("Number of templates: %d" % len(MyTemplateList))
c = 0
for temp in MyTemplateList:
    res = cv2.matchTemplate(img_gray, temp.template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.80
    loc = np.where(res >= threshold)
    while len(loc[0]) > 0:
        zipped = zip(*loc[::-1])
        for pt in zipped:
            s_img = temp.template
            l_img = img_gray
            x_offset = pt[0]
            y_offset = pt[1]
            l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img
            cv2.imshow("NJK", img_gray)
            cv2.waitKey()
            c += 1
            cv2.rectangle(img_rgb, pt, (pt[0] + temp.w, pt[1] + temp.h), (0, 0, 255), 2)
            cv2.putText(img_rgb, "%s" % c, (pt[0], pt[1] + temp.h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img_gray, (pt[0] + 10, pt[1] + 10), (pt[0] + temp.w - 10, pt[1] + temp.h - 10), (0, 0, 0), -1)
            break
        res = cv2.matchTemplate(img_gray, temp.template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.80
        loc = np.where(res >= threshold)

print(c, " teeth detected.")

cv2.imshow('res2.png', img_rgb)
cv2.waitKey(0)
cv2.imshow('res3.png', img_gray)
cv2.waitKey(0)
