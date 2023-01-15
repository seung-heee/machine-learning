import numpy as np
import cv2 as cv

def calcGrayHist(img):
    channels = [0]
    histSize = [256]
    histRange = [0, 256]

    hist = cv.calcHist([img], channels, None, histSize, histRange) # 히스토그램 구하기_numpy 배열로 리턴
    # img : 히스토그램을 찾을 이미지
    # channel : grayscale 이미지의 경우: 0, 컬러일 경우: 0 or 1 or 2
    # mask : 이미지 전체 : none
    # histSize : bin 개수
    # range : 픽셀값 범위, 보통 [0, 256]
    
    return hist

def getGrayHistImage(hist):
    _, histMax, _, _ = cv.minMaxLoc(hist) # hist의 최댓값 구함.

    imgHist = np.ones((100, 256), np.uint8) * 255 # 100행 256열 배열 생성, 모두 255(흰색)으로 초기화
    # np.ones(shape, dtype, order) # 1로 가득찬 배열 생성

    for x in range(imgHist.shape[1]): # imgHist.shape[1] : 256
      # 막대그래프 형태로 직접 imghist 행렬을 참조하여 영상을 생성함
        pt1 = (x, 100) # 시작 좌표
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax)) # 끝 좌표
        
        cv.line(imgHist, pt1, pt2, 0)
      # cv.line(이미지, 시작좌표, 끝좌표, 색상) : 두 좌표를 잇는 선 그리는 함수(각 빈에 대한 그래프 그리기)
      
      # imgHist.shape[0] : 100
      # imgHist.shape : (100, 256)
    return imgHist

def histgoram_stretching(): # 히스토그램 스트레칭
    src = cv.imread('hawkes.bmp', cv.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    gmin, gmax, _, _ = cv.minMaxLoc(src)

    dst = cv.convertScaleAbs(src, alpha=255.0/(gmax - gmin), beta=-gmin * 255.0/(gmax - gmin))

    cv.imshow('src', src)
    cv.imshow('srcHist', getGrayHistImage(calcGrayHist(src)))

    cv.imshow('dst', dst)
    cv.imshow('dstHist', getGrayHistImage(calcGrayHist(dst)))

    cv.waitKey()
    cv.destroyAllWindows()

def histgoram_equalization(): # 히스토그램 평활화
    src = cv.imread('hawkes.bmp', cv.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    dst = cv.equalizeHist(src) # 히스토그램 평활화를 수행하는 equalizeHist() 함수 제공

    cv.imshow('src', src)
    cv.imshow('srcHist', getGrayHistImage(calcGrayHist(src)))

    cv.imshow('dst', dst)
    cv.imshow('dstHist', getGrayHistImage(calcGrayHist(dst)))

    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
  histgoram_stretching()
  # histgoram_equalization()
