import numpy as np
import cv2 as cv


src = cv.imread('rose.bmp', cv.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    exit()

cv.imshow('src', src)

# 가우시안 필터의 표준 편차 sigma값을 1~6까지 증가시키며 언샤프마스크필터링 수행
for sigma in range(1, 6):
    # 가우시안 필터를 이용하영 구한 블러링 영상을 blurred에 저장
    blurred = cv.GaussianBlur(src, (0, 0), sigma) 

    alpha = 1.0
    dst = cv.addWeighted(src, 1 + alpha, blurred, -alpha, 0.0)
    # 언샤프 마스크 필터링 수행 scr1*alpha + scr2*beta + gamma
    # dst = cv.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
    # src1, src2 : 이미지 배열 / alpha, beta : 가중치 추가를 수행하는 동안 고려해야 할 가중치
    # gamma : 이미지의 모든 픽셀에 추가되는 정적 가중치
    
    desc = "sigma: %d" % sigma # 샤프닝 결과 영상 dst에 사용된 sigma 값을 출력
    # 이미지에 텍스트를 넣음
    #cv.putText(이미지배열, 설명, 위치, 사용할폰트, 폰트크기,폰트색상,폰트두께, 선유형)
    cv.putText(dst, desc, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
               1.0, 255, 1, cv.LINE_AA) #LINE_AA : 직선 그리기 함수

    cv.imshow('dst', dst)
    cv.waitKey()

cv.destroyAllWindows()
