import numpy as np
import cv2 as cv


def detect_face():
    src = cv.imread('kids.png')

    if src is None:
        print('Image load failed!')
        return

    classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    if classifier.empty():
        print('XML load failed!')
        return

    faces = classifier.detectMultiScale(src)

    for (x, y, w, h) in faces:
        cv.rectangle(src, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv.imshow('src', src)
    cv.waitKey()
    cv.destroyAllWindows()


def detect_eyes():
    src = cv.imread('kids.png')

    if src is None:
        print('Image load failed!')
        return

    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_classifier = cv.CascadeClassifier('haarcascade_eye.xml')

    if face_classifier.empty() or eye_classifier.empty():
        print('XML load failed!')
        return

    faces = face_classifier.detectMultiScale(src)

    for (x1, y1, w1, h1) in faces:
        cv.rectangle(src, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 255), 2)

        faceROI = src[y1:y1 + h1, x1:x1 + w1]
        eyes = eye_classifier.detectMultiScale(faceROI)

        for (x2, y2, w2, h2) in eyes:
            center = (int(x2 + w2 / 2), int(y2 + h2 / 2))
            cv.circle(faceROI, center, int(w2 / 2), (255, 0, 0), 2, cv.LINE_AA)

    cv.imshow('src', src)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    detect_face()
    detect_eyes()
