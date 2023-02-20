import numpy as np
import cv2 as cv


def affine_transform():
    src = cv.imread('tekapo.bmp')

    if src is None:
        print('Image load failed!')
        return

    rows = src.shape[0]
    cols = src.shape[1]

    src_pts = np.array([[0, 0],
                        [cols - 1, 0],
                        [cols - 1, rows - 1]]).astype(np.float32)
    dst_pts = np.array([[50, 50],
                        [cols - 100, 100],
                        [cols - 50, rows - 50]]).astype(np.float32)

    affine_mat = cv.getAffineTransform(src_pts, dst_pts)

    dst = cv.warpAffine(src, affine_mat, (0, 0))

    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()
def affine_translation():
    src = cv.imread('tekapo.bmp')

    if src is None:
        print('Image load failed!')
        return

    affine_mat = np.array([[1, 0, 150],
                           [0, 1, 100]]).astype(np.float32)

    dst = cv.warpAffine(src, affine_mat, (0, 0))

    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()
def affine_shear():
    src = cv.imread('tekapo.bmp')

    if src is None:
        print('Image load failed!')
        return

    rows = src.shape[0]
    cols = src.shape[1]

    mx = 0.3
    affine_mat = np.array([[1, mx, 0],
                           [0, 1, 0]]).astype(np.float32)

    dst = cv.warpAffine(src, affine_mat, (int(cols + rows * mx), rows))

    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()


def affine_scale(): # 크기
    src = cv.imread('rose.bmp')

    if src is None:
        print('Image load failed!')
        return

    # 다양한 보간법으로 확대
    dst1 = cv.resize(src, (0, 0), fx=4, fy=4, interpolation=cv.INTER_NEAREST) # 스케일 비율로 / 최근방 이웃 보간법
    dst2 = cv.resize(src, (1920, 1280)) # 기본값 linear / 효율, 퀄리티 좋음. 널리 사용됨 / 양선형 보간법
    dst3 = cv.resize(src, (1920, 1280), interpolation=cv.INTER_CUBIC) # 3차 보간법 4x4 
    dst4 = cv.resize(src, (1920, 1280), interpolation=cv.INTER_LANCZOS4) # 8x8 이웃 픽셀을 사용, 복잡하지만 퀄리티 좋음

    cv.imshow('src', src)
    # 400:800 좌표부터 500:900 크기 부분 영상을 출력
    cv.imshow('dst1', dst1[400:800, 500:900]) 
    cv.imshow('dst2', dst2[400:800, 500:900])
    cv.imshow('dst3', dst3[400:800, 500:900])
    cv.imshow('dst4', dst4[400:800, 500:900])
    cv.waitKey()
    cv.destroyAllWindows()

def affine_rotation(): # 회전
    src = cv.imread('tekapo.bmp')

    if src is None:
        print('Image load failed!')
        return

    cp = (src.shape[1] / 2, src.shape[0] / 2) # 영상 중심좌표
    affine_mat = cv.getRotationMatrix2D(cp, 90, 1) # 좌표 cp를 기준으로 반시계방향으로 20도 회전하는 행렬

    dst = cv.warpAffine(src, affine_mat, (0, 0)) # 행렬을 이용하여 어파인 변환

    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()

def affine_flip(): # 대칭
    src = cv.imread('eastsea.bmp')

    if src is None:
        print('Image load failed!')
        return

    cv.imshow('src', src)

    for flip_code in [1, 0, -1]: # 양수-좌우 / 0-상하 / 음수-상하좌우
        dst = cv.flip(src, flip_code) 

        desc = 'flipCode: %d' % flip_code
        cv.putText(dst, desc, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 0, 0), 1, cv.LINE_AA)

        cv.imshow('dst', dst)
        cv.waitKey()

    cv.destroyAllWindows()


if __name__ == '__main__':
    #affine_transform()
    #affine_translation()
    #affine_shear()
    #affine_scale() 
    #affine_rotation()
    affine_flip()
