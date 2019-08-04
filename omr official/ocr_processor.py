#import packages
from imutils.perspective import four_point_transform
from imutils import contours
from pdf2image import convert_from_path
import numpy as np
import argparse
import imutils
import cv2
import sys

filename = 'ocr_input.pdf'
i = 0
images = convert_from_path(filename)
for image in images:
    image.save("input_0"+str(i)+".png")
    my_image = cv2.imread("output"+str(i)+" .png")
    i += 1


def Sharpen(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    im = cv2.filter2D(img, -1, kernel)

    return im

def AutoRotate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols, _ = img.shape
    template = cv2.imread('S_1.jpg')

    w, h, _ = template.shape
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(Sharpen(gray), Sharpen(template), cv2.TM_CCOEFF_NORMED)
    threshold = 0.56
    loc = np.where(res >= threshold)
    point = []
    for pt in zip(*loc[::-1]):
        point = pt

    if (point[1] > rows / 2):
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst
    else:
        return img

def drawMinEnclose(resized,circles):
    (x,y),radius = cv2.minEnclosingCircle(circles)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(resized,center,radius,(0,255,0),2)


#read in file
image = cv2.imread("input_01.png")


ANSWER_KEY = {0: 0, 1: 0, 2: 0, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 0, 9: 0, 10: 2, 11: 4, 12: 0, 13: 3, 14: 1, 15: 1, 16: 4, 17: 0, 18: 3, 19: 1,
              20: 1, 21: 4, 22: 0, 23: 3, 24: 1, 25: 1, 26: 4, 27: 0, 28: 3, 29: 1, 30: 1, 31: 4, 32: 0, 33: 3, 34: 1, 35: 1, 36: 4, 37: 0, 38: 3, 39: 1,
              40: 1, 41: 4, 42: 0, 43: 3, 44: 1, 45: 1, 46: 4, 47: 0, 48: 3, 49: 1, 50: 1, 51: 4, 52: 0, 53: 3, 54: 1, 55: 1, 56: 4, 57: 0, 58: 3, 59: 1, 60: 3}

#image = cv2.resize(imgFile,(500,500))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(blurred, 75, 200)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure that at least one contour was found
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break

paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# apply Otsu's thresholding method to binarize the warped
# piece of paper
blurred = cv2.GaussianBlur(warped, (3, 3), 0)
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations=2)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(len(cnts))
questionCnts = []
markOutOf30 = True


for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    if w >= 20 and h >= 20 and (x >= 450 and x <= 675) or (x >= 750 and x <= 975):
        cv2.drawContours(paper, [c], -1, (0, 255, 0), 2)
        #drawMinEnclose(paper, c)
        questionCnts.append(c)


questionCnts = contours.sort_contours(questionCnts,	method="top-to-bottom")[0]
correct = 0
#print(questionCnts[25])
#grade
#answer grid A
answer_grid_a_len = 30

completeQuestionsCnts = []

for i in range( 0 , 50 , 1):
    (x,y,w,h) = cv2.boundingRect(questionCnts[i])
    a = x >= 450 and x <= 480
    b = x >= 480 and x <= 510
    c = x >= 510 and x <= 540
    d = x >= 540 and x <= 570
    e = x >= 570 and x <= 600

    a1 = x >= 800 and x <= 830
    b1 = x >= 830 and x <= 860
    c1 = x >= 860 and x <= 890
    d1 = x >= 890 and x <= 920
    e1 = x >= 920 and x <= 950

    if a or a1:
        print("question ", i + 1, " ans: A")
        if (ANSWER_KEY[i] == 0):
            correct += 1
            drawMinEnclose(paper, questionCnts[i])
        else:
            cv2.drawContours(paper, [questionCnts[i]], -1, (0, 0, 255), 2)
    elif b or b1:
        print("question ", i + 1, " ans: B")
        if (ANSWER_KEY[i] == 1):
            correct += 1
            drawMinEnclose(paper, questionCnts[i])
        else:
            cv2.drawContours(paper, [questionCnts[i]], -1, (0, 0, 255), 2)
    elif c or c1:
        print("question ", i + 1, " ans: C")
        if (ANSWER_KEY[i] == 2):
            correct += 1
            drawMinEnclose(paper, questionCnts[i])
        else:
            cv2.drawContours(paper, [questionCnts[i]], -1, (0, 0, 255), 2)
    elif d or d1:
        print("question ", i + 1, " ans: D")
        if (ANSWER_KEY[i] == 3):
            correct += 1
            drawMinEnclose(paper, questionCnts[i])
        else:
            cv2.drawContours(paper, [questionCnts[i]], -1, (0, 0, 255), 2)
    elif e or e1:
        print("question ", i + 1, " ans: E")
        if (ANSWER_KEY[i] == 4):
            correct += 1
            drawMinEnclose(paper, questionCnts[i])
        else:
            cv2.drawContours(paper, [questionCnts[i]], -1, (0, 0, 255), 2)

# grab the test taker
if markOutOf30:
    score = (correct / 30.0) * 100
else:
    score = (correct / 60.0 ) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("temp2", paper)
cv2.waitKey(0)