import numpy as np
import cv2 as cv
import random

# 가우시안 잡음 모델
def noise_gaussian():
    src = cv.imread('lenna.bmp', cv.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    cv.imshow('src', src)
    
    # 표준 편차가 10, 20, 30이 되도록 반복문 수행
    for stddev in [10, 20, 30]:
        # 정수형을 사용하는 영상 크기만큼의 배열을 생성
        noise = np.zeros(src.shape, np.int32)
        # 평균이 0이고 표준편차가 stddev인 가우시안 잡음 생성
        # noise 배열에 저장
        cv.randn(noise, 0, stddev)

        # 입력 영상에 가우시안 잡음(noise)를 더하여 결과영상 dst 생성, 영상의 깊이 CV_8U
        dst = cv.add(src, noise, dtype=cv.CV_8UC1)

        desc = 'stddev = %d' % stddev # 설명
        #cv.putText(잡음이 추가된 영상, 설명, 위치, 사용할폰트, 폰트크기,폰트색상,폰트두께, 선유형)
        cv.putText(dst, desc, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, 255, 1, cv.LINE_AA) # 이미지에 텍스트를 넣음
        
        cv.imshow('dst', dst)
        cv.waitKey()

    cv.destroyAllWindows()


# 양방향 필터
def filter_bilateral():
    src = cv.imread('lenna.bmp', cv.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    # 가우시안 잡음 추가 src
    noise = np.zeros(src.shape, np.int32) #입력영상의 크기만큼 정수형 배열 생성
    cv.randn(noise, 0, 5) # 평균0, 표준편차5인 가우시안 필터링을 noise에 저장
    cv.add(src, noise, src, dtype=cv.CV_8UC1) # 입력영상(src)에 가우시안 필터링(noise)을 더함

    dst1 = cv.GaussianBlur(src, (0, 0), 5) # 표준 편차가 5인 가우시안 필터링 수행
    dst2 = cv.bilateralFilter(src, -1, 10, 5) #양방향 필터링
    # 색공간 표준편차 10 / 좌표공간 표준편차 5를 사용하는 양방향 필터링 수행
    
    cv.imshow('src', src) # 평균0, 표준편차5인 가우시안 잡음이 추가된 영상
    cv.imshow('dst1', dst1) # 편차가 5인 가우시안 필터링 > 잡음은 줄었지만, 경계부분이 함께 블러링
    cv.imshow('dst2', dst2) # 양방향 필터 적용 : 사물의 경계는 그대로 유지
    # 평탄한 영역의 잡음은 크게 줄어 눈으로 보기에 매우 깔끔한 느낌
    cv.waitKey()
    cv.destroyAllWindows()

# 미디언 필터
def filter_median():
    src = cv.imread('lenna.bmp', cv.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return
    # 영상에서 10%에 해당하는 픽셀 값을 0 or 255로 설정
    for i in range(0, int(src.size / 10)):
        x = random.randint(0, src.shape[1] - 1)
        y = random.randint(0, src.shape[0] - 1)
        src[x, y] = (i % 2) * 255

    dst1 = cv.GaussianBlur(src, (0, 0), 1)
    dst2 = cv.medianBlur(src, 3) 

    cv.imshow('src', src)
    cv.imshow('dst1', dst1) # 가우시안 필터
    cv.imshow('dst2', dst2) # 미디어 필터
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    #noise_gaussian()
    #filter_bilateral()
    filter_median()
