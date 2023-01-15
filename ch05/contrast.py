import numpy as np
import cv2 as cv


def contrast1(): # 기본적인 명암비 조절
    src = cv.imread('lenna.bmp', cv.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    s = 2
    dst = cv.multiply(src, s) # 모든 픽셀값에 일정 상수를 곱함

    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()


def contrast2(): # 효과적인 명암비 조절 방법
    src = cv.imread('lenna.bmp', cv.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    alpha = 1 # 명암비, 기울기
    dst = cv.convertScaleAbs(src, alpha=1+alpha, beta=-128*alpha)
    # 기준값 128을 이용해 효과적으로 픽셀값을 조절
    
    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    contrast1()
    contrast2()
