import numpy as np
import cv2 as cv


def calcGrayHist(img):
    channels = [0]
    histSize = [256]
    histRange = [0, 256]

    hist = cv.calcHist([img], channels, None, histSize, histRange)

    return hist


def getGrayHistImage(hist):
    _, histMax, _, _ = cv.minMaxLoc(hist)

    imgHist = np.ones((100, 256), np.uint8) * 255
    for x in range(imgHist.shape[1]):
        pt1 = (x, 100)
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))
        cv.line(imgHist, pt1, pt2, 0)

    return imgHist


src = cv.imread('hawkes.bmp', cv.IMREAD_GRAYSCALE)
hist = calcGrayHist(src)
hist_img = getGrayHistImage(hist)

cv.imshow('src', src)
cv.imshow('srcHist', hist_img)
cv.waitKey()
cv.destroyAllWindows()