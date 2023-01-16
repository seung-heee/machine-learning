import numpy as np
import cv2 as cv

def calcGrayHist(img): # 히스토그램 구하기
    channels = [0] # grayscale 이미지
    histSize = [256] # bin 개수
    histRange = [0, 256] # 픽셀 값 범위

    hist = cv.calcHist([img], channels, None, histSize, histRange) # 히스토그램 계산기_numpy 배열로 리턴
    # img : 이미지 배열
    # channel : 분석할 컬러 / grayscale 이미지: 0, 컬러: 0 or 1 or 2
    # mask : 분석할 영역 / 이미지 전체 -> none
    # histSize : 히스토그램 크기, bin(x축 값) 개수
    # range : 픽셀 값 범위(x축 범위), 보통 [0, 256]
    
    # 이미지를 넣고 그레이스케일의 컬러를 분석한다.
    # 영역은 전체이면서 x축 값은 256개로 0부터 255까지 지정한다.
    
    return hist

def getGrayHistImage(hist): # 그레이스케일 영상의 히스토그램 그래프 그리기
    _, histMax, _, _ = cv.minMaxLoc(hist) # hist의 최댓값 구함.

    imgHist = np.ones((100, 256), np.uint8) * 255 # 100행 256열 배열 생성, 모두 255(흰색)으로 초기화
    # np.ones(shape, dtype, order) # 1로 가득찬 배열 생성

    for x in range(imgHist.shape[1]): 
      # 막대그래프 형태로 직접 imghist 행렬을 참조하여 영상을 생성함
        pt1 = (x, 100) # 시작 좌표
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax)) # 끝 좌표
        
        cv.line(imgHist, pt1, pt2, 0)
      # cv.line(이미지, 시작좌표, 끝좌표, 색상) : 두 좌표를 잇는 선 그리는 함수(각 빈에 대한 그래프 그리기)
      
      # imgHist.shape : (100, 256)
      # imgHist.shape[0] : 100
      # imgHist.shape[1] : 256

    return imgHist

def histgoram_stretching(): # 히스토그램 스트레칭 / 특정 영역에 몰려 있는 경우, 그레이스케일 전 구간에서 걸쳐 나타나도록 변경
    src = cv.imread('hawkes.bmp', cv.IMREAD_GRAYSCALE) # 그레이스케일로 이미지 입력

    
    if src is None:
        print('Image load failed!')
        return

    gmin, gmax, _, _ = cv.minMaxLoc(src) # 최소, 최대 포인터 구함

    dst = cv.convertScaleAbs(src, alpha = 255.0/(gmax - gmin), beta=-gmin * 255.0/(gmax - gmin)) # 정규화 
    # alpha : 직선의 기울기 / beta : y절편 => 기울기와, y절편을 이용해 직선의 방정식을 구해 (gmin, 0)과 (gmax, 255)를 지나가게 한다. (그래프 b)

    
    cv.imshow('src', src)
    cv.imshow('srcHist', getGrayHistImage(calcGrayHist(src))) # 원본 

    cv.imshow('dst', dst)
    cv.imshow('dstHist', getGrayHistImage(calcGrayHist(dst))) # 변형 결과

    cv.waitKey()
    cv.destroyAllWindows()

def histgoram_equalization(): # 히스토그램 평활화 / 픽셀 값들을 평준화 시키면 그만큼 선명도가 올라감
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