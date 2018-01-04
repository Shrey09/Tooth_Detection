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

print(len(MyTemplateList))
#template = cv2.imread('coin.jpg',0)
#w, h = template.shape[::-1]
c=0
for temp in MyTemplateList:
    res = cv2.matchTemplate(img_gray,temp.template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + temp.w, pt[1] + temp.h), (0,0,255), 2)
        c+=1
print(c," teeth detected.")

cv2.imwrite('res2.png',img_rgb)