> # 2021 Capstone Design

# 개요

- 딥러닝과 머신 비전을 활용한 실시간 물체 인식 프로그램
- 물체 이동을 위한 컨베이어 시스템 구축
- 협동로봇 제어를 통해 물체를 Pick&Place 후 종류별로 분류

# 팀 구성
## HW
- 김** - 컨베이어 해석 및 제작
- 김** - 컨베이어 설계 및 모터 제어

## SW
- 안경준(팀장) - 물체 인식프로그램 개발, 딥러닝 모델 모델링, 프로그램 통합
- 이** - 협동로봇 제어 프로그램 개발

# 진행

## 계통도
</br>
<p align="center"><img src=./src/system.png title="System"></p>
</br>

## HW
### 컨베이어
#### 개요
- 목표 속도 = `1 m/min`
- 물성치
    - 탄성계수 : 200 GPa
    - 푸아송비 : 0.3
    - 밀도 : 7850 kg/m^3
    - 허용응력 : 400 MPa(검증)

### 카메라 모듈
- See3Cam_Cu135 module 사용
    - 노출도 조절을 통해 프레임 확보
    - 다양한 해상도 지원
    - 세부 세팅을 통해 적절한 인식환경 조성

### 협동 로봇
- UR3 로봇 팔 + RG2 Gripper 사용
- TCP/IP 프로토콜의 소켓 통신 사용
- `Python` 제어

## SW
### 개발환경
- Python
    - Jupyter Notebook
        - TensorFlow
    - Spyder
        - OpenCV

### 흐름도
</br>
<p align="center"><img src=./src/flowchart.png title="flowchart"></p>
</br>

### 딥러닝
- 과대 적합 방지
    - 데이터 증강(회전, 확대/축소)
    - 이미지 해상도 조절
    - 데이터 학습량 조절(배치 크기 조절)
    - 학습 횟수 조절
    - Dropout 사용
- 일반적인 CNN 모델 설계
</br>
<p align="center"><img src=./src/model.png title="CNN model"></p>
</br>
- 학습한 가중치 저장 및 분류에 재사용
</br>
<p align="center"><img src=./src/sample.png title="sample result"></p>
</br>

### Contour 검출
- 구조
    - 원본 이미지 이진화
    - `OpenCV` `contour` 메서드를 사용하여 Contour 검출
    - 검출한 Contour 근사
    - 근사시킨 Contour의 중심을 계산하여 추출

</br>
<p align='center'>
    <span><img src=./src/contour1.png width='40%' title="contour"></span>
    <span><img src=./src/contour2.png width='40%' title="approximated contour"></span>
</p>
</br>

- 문제
    - 이진화된 이미지가 구분이 안되는 경우 노이즈를 검출하는 현상 발생
        - 실제 환경에서는 조명이 균일하지 않아 이미지의 밝기가 균일하지 않음
        - 반사, 잔상, 그림자 등의 요소들이 노이즈를 생성
    - `cv2.adpativeThreshold` 함수를 통해 이미지의 구역마다 임계값(이진화의 기준)을 달리 설정해 해결하려 했으나 정확도가 떨어짐
</br>
<p align='center'>
    <span><img src=./src/noise.png width='40%' title="noise"></span>
    <span><img src=./src/binary.png width='40%' title="binary"></span>
</p>
</br>

- 해결
    - `cv2.Canny` 함수를 사용하여 노이즈를 최대한 줄임
        - 하지만 환경, 피사체마다 적절한 임계값 설정이 필요함
    - 암막 설치를 통해 일정한 임계값으로도 문제 해결 가능

### Line 검출
- `Canny Edge`를 통해 Edge 이미지 추출
- `Hough` 변환 알고리즘을 사용하여 Edge 이미지의 직선 추출
- 직선 위의 모든 점을 사용하여 각도를 계산하고 각도의 평균을 통해 직선의 각도를 추출

</br>
<p align="center"><img src=./src/cannyedge.png title="cannyedge"></p>
</br>

### 결과 이미지
- 이미지에 실제 거리를 반영하여 이미지의 중심 좌표를 기준으로 UR3의 위치와 이동 거리를 설정

</br>
<p align="center"><img src=./src/result.png title="result"></p>
</br>

</br>
<p align="center"><img src=./src/lego_sort_result.gif width='50%' title="result"></p>
</br>
