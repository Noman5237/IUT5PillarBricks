# Template Matching
# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv.imread('editedbrick.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread('sample2.jpg', 0)
template = cv.imread('sample1.jpg',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.5

while threshold > 0:
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv.imwrite('res10' + str(int(threshold * 10)) + '.jpg',img_rgb)
    print("res10" + str(int(threshold * 10)) + '.jpg')
    threshold -= 0.1
