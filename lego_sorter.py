# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:40:12 2021

@author: An Gyeong Jun
"""

# 필요 라이브러리 호출
import tensorflow as tf
import cv2
import numpy as np
import math
import serial
import socket
import time
import onrobot_rg2_gripper
import urx
import logging

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 딥러닝 모델(model3_checkpoint 가중치 학습 환경과 똑같이 설정)
data_dir = "C:/Users/user/tensorflow/sort2"
categories = ['color_link','gear', 'link']
nb_class=len(categories)
batch_size = 36
img_height = 180
img_width = 240

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

num_classes=3

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.3),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# 학습한 가중치 불러오기
model.load_weights('model3_checkpoint')

# UR3 소켓 통신 설정 및 시작
HOST = "169.254.162.52"   # The remote host
PORT = 30002              # The same port as used by the server
print ("Starting Program")
logging.basicConfig(level=logging.INFO)
rob = urx.Robot("169.254.162.52")
gripper = onrobot_rg2_gripper.OnRobotGripperRG2(rob)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))


# Atmega128 UART 시리얼 통신 시작(포트'COM5')
ser=serial.Serial('COM5', baudrate = 57600, timeout=1)
readed = ser.read(1)
# 카메라 연결, 해상도 불러오기
cap = cv2.VideoCapture(2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#각도 기본값 설정
w=90
x = math.radians(w)


#프로그램 반복
while(True):

    # 카메라 화면 프레임 단위로 불러오기(카메라 영상 확인)
    ret, img_color = cap.read()
    # 카메라 불러오기 실패시 프로그램 종료
    if ret == False:
        break

    # Atmega128 값 존재여부 확인
    wt = ser.inWaiting()

    # Atmega128 값 있을시 값 읽기(카메라 영상 프레임 확보를 위한 조건)
    if wt > 0:
        readed=ser.read(1)
        
    # ROI 설정(인식영역 설정)    
    img_input = img_color.copy()
    cv2.rectangle(img_color, (80, 60), (width-80, height-60), (0, 0, 255), 3)
    cv2.imshow('bgr', img_color)
    img_roi = img_input[60:height-60, 80:width-80]

    # 키입력 변수 설정
    key = cv2.waitKey(1)
    
    # Esc 누를 시 컨베이어 벨트 정지 및 시리얼 통신 종료, UR3 소켓 통신 종료 후 프로그램 종료
    if key == 27:
        op='5'
        ser.write(op.encode())
        ser.close()        
        print("통신이 종료되었습니다")
        data = s.recv(1024)
        s.close()
        print ("Received", repr(data))
        print ("Program finish")
        break
    
    # keyboard 1키 = 컨베이어 벨트 속도 증가
    if key == 49:
        op='1'
        ser.write(op.encode())
    # keyboard 2키 = 컨베이어 벨트 속도 감소
    if key == 50:
        op='2'
        ser.write(op.encode())
    # keyboard 3키 = 컨베이어 벨트 정방향 재생        
    if key == 51:
        op='3'
        ser.write(op.encode())
    # keyboard 4키 = 컨베이어 벨트 역방향 재생        
    if key == 52:
        op='4'
        ser.write(op.encode())
    # keyboard 5키 = 컨베이어 벨트 정지        
    if key == 53:
        op='5'
        ser.write(op.encode())
    # keyboard 6키 = LED 전원 on        
    if key == 54:
        op='6'
        ser.write(op.encode())        
    # keyboard 7키 = LED 전원 off        
    if key == 55:
        op='7'
        ser.write(op.encode())
        
    # 센서1 인식 시 인식 및 중심좌표 추출 시작
    if readed == b'U':
        # 잔상 및 노이즈 제거를 위한 딜레이 설정
        time.sleep(1)
        # ROI 영역 잘라서 인식
        ret, img_color = cap.read()
        img_input = img_color.copy()
        img_roi = img_input[60:height-60, 80:width-80]
        img=cv2.resize(img_roi, (240, 180), interpolation=cv2.INTER_AREA)
        img2=cv2.resize(img_roi, (480, 360), interpolation=cv2.INTER_AREA)
        predictions = model.predict(img[np.newaxis,:])
        score = tf.nn.softmax(predictions[0])
        # 인식 결과 출력
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
            )
        # 인식 결과 카테고리 변수 설정
        cln = class_names[np.argmax(score)]
        # 인식 결과가 'gear'일 경우 테두리 검출 후 중심좌표 추출
        if cln == 'gear':
            # Canny Edge Detection(환경마다 설정 필요)
            img2=cv2.resize(img_roi, (480, 360), interpolation=cv2.INTER_AREA)
            imgray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            th = cv2.Canny(imgray, 1000,100)
            # 가장자리 컨투어 찾기
            contours, hierachy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = contours[0]
            # 전체 둘레의 0.01로 오차 범위 지정
            epsilon = 0.01 * cv2.arcLength(contour, True)
            # 근사 컨투어 계산 ---③
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # 중심좌표 계산
            M = cv2.moments(approx)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # 중심좌표 출력
            cv2.circle(img2, (cx, cy), 5, (0,255,255), -1)
            print(cx,cy)
        
        # 인식 결과가 'link'이거나 'color_link'일 경우 테두리 검출 후 중심좌표 추출 및 Hough변환으로 선 검출 후 각도 추출
        if cln == 'link' or cln == 'color_link':
            # Canny Edge Detection(환경마다 설정 필요)
            imgray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
            th = cv2.Canny(imgray, 500,150)
            # 가장자리 컨투어 찾기
            contours, hierachy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = contours[0]
            # 전체 둘레의 0.01로 오차 범위 지정
            epsilon = 0.01 * cv2.arcLength(contour, True)
            # 근사 컨투어 계산
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # 중심좌표 계산
            M = cv2.moments(approx)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # 중심좌표 출력
            cv2.circle(img2, (cx, cy), 5, (255,0,255), -1)
            print(cx,cy)
            # 확률적 Hough변환을 이용한 선 검출
            lines = cv2.HoughLinesP(th, 1, np.pi/180, 10, None, 20, 2)
            # 각도 변수 설정
            angles = []
            # 선 각도 계산
            for x1, y1, x2, y2 in lines[0]:
                cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 3)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
            # 각도 출력
            median_angle = np.median(angles)
            print ("Angle is {}".format(median_angle))
            # UR3 gripper 각도 설정
            w = median_angle + 90
            if w > 180:
                w -= 180
            x = math.radians(w)
            
        # 테두리, 중심좌표, 선 출력 창
        cv2.drawContours(img2, [approx], 0, (0,255,0), 3)
        cv2.imshow('contour', img2)
        
        # UR3 초기 위치 설정
        gripper.open_gripper(target_width=80,target_force=40, payload=0.78,set_payload=False,depth_compensation=False,slave=False, wait=0)
        time.sleep(1)   # first gripper
        s.send ("movej ([-2.3846433569, -3.222052332, 1.1718140597, -2.6764624079, 1.578301242, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
        time.sleep(1)   # first position
        # Atmega 값 초기화
        readed = 0
        # UR3 무한루프 방지
        count = 0
        
    # 센서2 인식 시 컨베이어 정지 및 Pick&Place 진행
    if readed == b'P':
        # UR3가 Pick&Place 한 번 진행 후 루프 해제(무한루프 방지)
        while(count<1):
            # 'link' 이거나 'color_link'일 때
            if cln=='link' or cln=='color_link':
                # pick position
                if cy<75:
                    s.send (f"movej ([-2.381152698, -3.313856650, 0.8857545953, -2.298773157, 1.5762068474, {x}], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 480mm, link pick position
                elif 75<=cy<105:
                    s.send (f"movej ([-2.382025363, -3.350683097, 0.9786061115, -2.355321825, 1.5762068474, {x}], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 470mm, link pick position
                elif 105<=cy<135:
                    s.send (f"movej ([-2.382723494, -3.399901382, 1.1065387457, -2.434036174, 1.5765559133, {x}], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode()) 
                    time.sleep(5)   # y = 455mm, link pick position
                elif 135<=cy<165:
                    s.send (f"movej ([-2.383421626, -3.442312883, 1.2222540751, -2.507340003, 1.5770795121, {x}], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode()) 
                    time.sleep(5)   # y = 440mm, link pick position
                elif 165<=cy<195:
                    s.send (f"movej ([-2.384294291, -3.479488396, 1.3287191595, -2.576804107, 1.5774285779, {x}], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 425mm, link pick position
                elif 195<=cy<225:
                    s.send (f"movej ([-2.385166955, -3.511951520, 1.4282029269, -2.643650217, 1.5779521767, {x}], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode()) 
                    time.sleep(5)   # y = 410mm, link pick position
                elif 225<=cy<255:
                    s.send (f"movej ([-2.386214153, -3.540574920, 1.5219271077, -2.708576466, 1.5784757755, {x}], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode()) 
                    time.sleep(5)   # y = 395mm, link pick position
                elif 255<=cy<285:
                    s.send (f"movej ([-2.387435883, -3.565533128, 1.6107643666, -2.772455516, 1.5789993742, {x}], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode()) 
                    time.sleep(5)   # y = 380mm, link pick position
                elif 285<=cy:
                    s.send (f"movej ([-2.388134015, -3.579844828, 1.667313034, -2.814867017, 1.579348440, {x}], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode()) 
                    time.sleep(5)   # y = 370mm, link pick position
                gripper.open_gripper(target_width=5,target_force=40, payload=0.78,set_payload=False,depth_compensation=False,slave=False, wait=0)
                time.sleep(2)   # second gripper
                s.send ("movej ([-2.3846433569, -3.222052332, 1.1718140597, -2.6764624079, 1.578301242, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                time.sleep(5)   # first position
                
                # 'link' place position
                if cln=='link':
                    s.send ("movej ([-0.9475392509, -3.113492852, 0.8414232323, -2.4537583953, 1.577952176, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # first place position(link)
                    s.send ("movej ([-0.9450957899, -3.856130449, 1.0967649019, -1.9664624682, 1.577952176, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(3)   # second place position(link)
                    gripper.open_gripper(target_width=80,target_force=40, payload=0.78,set_payload=False,depth_compensation=False,slave=False, wait=0)
                    time.sleep(2)   # first gripper
                    s.send ("movej ([-0.9475392509, -3.113492852, 0.8414232323, -2.4537583953, 1.577952176, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(3)   # first place position(link)
                    s.send ("movej ([-2.3846433569, -3.222052332, 1.1718140597, -2.6764624079, 1.578301242, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # first position
                    count += 1
                
                # 'color_link' place position
                elif cln=='color_link':
                    s.send ("movej ([-0.5880014249, -3.221703266, 1.1625638147, -2.6579619178, 1.582839098, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # first place position(color_link)
                    s.send ("movej ([-0.5857324969, -4.007625028, 1.3782865102, -2.0879373841, 1.582839098, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(3)   # second place position(color_link)
                    gripper.open_gripper(target_width=80,target_force=40, payload=0.78,set_payload=False,depth_compensation=False,slave=False, wait=0)
                    time.sleep(2)   # first gripper
                    s.send ("movej ([-0.5880014249, -3.221703266, 1.1625638147, -2.6579619178, 1.582839098, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(3)   # first place position(color_link)
                    s.send ("movej ([-2.3846433569, -3.222052332, 1.1718140597, -2.6764624079, 1.578301242, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # first position
                    count += 1
                
            # 'gear'일 때
            if cln=='gear':
                # pick position
                if cy<75:
                    s.send ("movej ([-2.381152698, -3.323630494, 0.892386846, -2.295806098, 1.5762068474, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode()) 
                    time.sleep(5)   # y = 480mm, gear pick position
                elif 75<=cy<105:
                    s.send ("movej ([-2.381850830, -3.360456941, 0.9843656981, -2.351656634, 1.5760323145, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 460mm, gear pick position
                elif 105<=cy<135:
                    s.send ("movej ([-2.382548961, -3.409675226, 1.1121237993, -2.430196450, 1.5765559133, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 425mm, gear pick position
                elif 135<=cy<165:
                    s.send ("movej ([-2.383247093, -3.452435793, 1.2278391287, -2.503151213, 1.5769049791, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 390mm, gear pick position
                elif 165<=cy<195:
                    s.send ("movej ([-2.384119758, -3.489785839, 1.3341296802, -2.572091718, 1.5774285779, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 460mm, gear pick position
                elif 195<=cy<225:
                    s.send ("movej ([-2.384992422, -3.522598029, 1.4334389146, -2.638414230, 1.5779521767, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 425mm, gear pick position
                elif 225<=cy<255:
                    s.send ("movej ([-2.386039620, -3.551395961, 1.5268140296, -2.702991412, 1.5784757755, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 390mm, gear pick position
                elif 255<=cy<285:
                    s.send ("movej ([-2.387261350, -3.576528703, 1.6158258214, -2.766870463, 1.5789993742, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                    time.sleep(5)   # y = 390mm, gear pick position
                elif 285<=cy:
                    s.send ("movej ([-2.388134015, -3.591364001, 1.672723555, -2.808758365, 1.579348440, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode()) 
                    time.sleep(5)   # y = 370mm, gear pick position
                gripper.open_gripper(target_width=5,target_force=40, payload=0.78,set_payload=False,depth_compensation=False,slave=False, wait=0)
                time.sleep(2)   # second gripper
                s.send ("movej ([-2.3846433569, -3.222052332, 1.1718140597, -2.6764624079, 1.578301242, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                time.sleep(5)   # first position
        
                # 'gear' place position
                s.send ("movej ([-0.2462659574, -3.178244567, 1.0159561575, -2.5455627140, 1.582490032, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                time.sleep(5)   # first place position(gear)
                s.send ("movej ([-0.2441715623, -3.935368397, 1.2400564335, -2.0125391604, 1.582490032, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                time.sleep(3)   # second place position(gear)
                gripper.open_gripper(target_width=80,target_force=40, payload=0.78,set_payload=False,depth_compensation=False,slave=False, wait=0)
                time.sleep(2)   # second gripper
                s.send ("movej ([-0.2462659574, -3.178244567, 1.0159561575, -2.5455627140, 1.582490032, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                time.sleep(3)   # first place position(gear)
                s.send ("movej ([-2.3846433569, -3.222052332, 1.1718140597, -2.6764624079, 1.578301242, 1.570796326], a = 1.3962634015954636, v = 1.0471975511965976)".encode()+"\n".encode())
                time.sleep(5)   # first position
                count += 1
            
            # pick&place 완료 후 컨베이어 벨트 정방향 작동
            op = '3'
            ser.write(op.encode())
            # Atmega128 값 초기화
            readed = 0


cap.release()
cv2.destroyAllWindows()