from matplotlib import pyplot as plt
import sys
import cv2


# 이미지 읽고 gray scale 로 변환
img = cv2.imread("Test1.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scale_factor = 0.5
resized_img_gray = cv2.resize(img_gray, None, fx=scale_factor, fy=scale_factor)
cv2.imshow('result', resized_img_gray)
cv2.waitKey(0)


# cascade classifier 읽고 얼굴 이미지 인식
cascade_file = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if cascade_file.empty():
    print("xml load failed")
    sys.exit()

face_list = cascade_file.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=15)
# img_gray : 검출 대상이 그레이스케일 이미지라고 명시
# scaleFactor : 얼굴 크기가 다양한 이미지를 처리할 수 있도록 입력 이미지를 계속 축소검색 - 현재는 20%씩 축소
# minNeighbors : 후보 영역이 얼굴로 판단되기 위해 최소한 몇 개의 "이웃" 검출이 필요한지를 지정합니다



# 인식한 얼굴에 사각형 표시
color = (255, 0, 0) # 색상은 파란색 - (B, G, R)
for face in face_list: 
      # face_list에는 검출된 얼굴의 정보가 들어있다
	x, y, w, h = face 
      # 얼굴이 (x, y)좌표에서 너비는 w, 높이는 h에 구역에 위치해있다는 뜻
	cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
      # img원본에다 사각형을 그린다 
      # (x, y):사각형의 왼쪽 상단 모서리 좌표
      # (x+w, y+h): 사각형의 오른쪽 하단 모서리 좌표
      # 색깔은 파란색
      # 두께는 2


scale_factor = 0.5  # 축소 비율
resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
# 축소 비율만큼 사진을 축소해서 보여준다
#resized_img = cv2.resize(img, (200, 300)) 뒤에 fx,fy는 무시된다. 200*300으로 보여준다.



cv2.imshow('result', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
