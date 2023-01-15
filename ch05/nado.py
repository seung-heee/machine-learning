import cv2 as cv

def img_print(): # 이미지 출력 
  img_color = cv.imread('lenna.bmp', cv.IMREAD_COLOR) 
  img_gray = cv.imread('lenna.bmp', cv.IMREAD_GRAYSCALE) 
  img_unchanged = cv.imread('lenna.bmp', cv.IMREAD_UNCHANGED)

  cv.imshow('img_color', img_color) 
  cv.imshow('img_gray', img_gray) 
  cv.imshow('img_unchanged', img_unchanged) 

  cv.waitKey(0)
  cv.destroyAllWindows()

  img = cv.imread('lenna.bmp')
  print(img.shape)

def video_print(): # 비디오 출력
  cap = cv.VideoCapture('video.mp4')
  
  while cap.isOpened(): # 동영상 파일이 올바로 열렸다면
    ret, frame = cap.read() # ret : 성공 여부, frame : 받아온 이미지(프레임)
    if not ret:
      print('더 이상 가져올 프레임이 없어요')
      break
    
    cv.imshow('video', frame)
  
    if cv.waitKey(25) == ord('q'):
      print('사용자 입력에 의해 종료합니다.')
      break
  
  cap.release() # 자원 해제
  cv.destroyAllWindows() # 모든 창 닫기
  
def camera(): # 카메라 출력
  cap = cv.VideoCapture(0) # 0번째 카메라 장치
  
  if not cap.isOpened(): # 카메라가 잘 열리지 않은 경우
    exit() # 프로그램 종료
    
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    
    cv.imshow('camera', frame)
    if cv.waitKey(1) == ord('q'):
      break
    
  cap.release()
  cv.destroyAllWindows()

img = cv.imread('book.jpg', cv.IMREAD_GRAYSCALE)


cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
  

