import cv2
import numpy as np
import glob

class MyObj():
    template = None
    w = None
    h = None


img_rgb = cv2.imread('Target.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

files = glob.glob("./Tooth_images/*")

MyTemplateList = []

for pth in files:
    temp = MyObj()
    temp.template = cv2.imread(pth,0)
    w, h = temp.template.shape[::-1]
    temp.w = w
    temp.h = h
    MyTemplateList.append(temp)
    #print("W: ",w," H: ",h)

print(len(MyTemplateList))
#template = cv2.imread('coin.jpg',0)
#w, h = template.shape[::-1]
c=0
for temp in MyTemplateList:
    res = cv2.matchTemplate(img_gray,temp.template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.80
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        #print("X: ",pt[0]," Y: ",pt[1])
        c += 1
        ##deleteing from gray scaled img
        #using text
        #cv2.putText(img_gray,"%s"%c , (pt[0],pt[1]+h) ,cv2.FONT_HERSHEY_COMPLEX,1, (0, 0, 255), 1,cv2.LINE_AA)
        cv2.putText(img_rgb,"%s"%c , (pt[0],pt[1]+h) , cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

        #using rectangles
        cv2.rectangle(img_gray, pt, (pt[0] + temp.w, pt[1] + temp.h), (0, 0, 255), -1)
        cv2.imwrite('/home/nilesh/PycharmProjects/ToothDetection/res3.png', img_gray)
        img_gray = cv2.cvtColor(cv2.imread('/home/nilesh/PycharmProjects/ToothDetection/res3.jpg'), cv2.COLOR_BGR2GRAY)

        ##Main output with rectangles
        cv2.rectangle(img_rgb, pt, (pt[0] + temp.w, pt[1] + temp.h), (0,0,255), 2)

print(c," teeth detected.")
cv2.imwrite('res2.png',img_rgb)
cv2.imwrite('res3.png',img_gray)