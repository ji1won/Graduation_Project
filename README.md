# Graduation_Project : 바른생활 생활바름
**인공지능 모델을 이용한 바른자세 유지 도움 프로그램**

### 🔸**개요**

현대사회에서는 노트북과 컴퓨터 앞에서 공부를 하거나 업무를 하는 시간은 점점 늘어나고 있습니다. 그래서 무의식 중에 자세가 흐트러지기 쉬운 공부중, 업무중일 때 바른자세를 유지할 수 있게 도와주는 프로그램을 제작하고자 하였습니다.

### 🔸작품 개발 목적

1. **자세 인식 및 교정**

카메라를 통해 사용자 자세를 분류하고, 잘못된 자세일 경우 경고 메시지 알림음과 알림창으로 사용자가 잘못된 자세를 유지 중임을 인지하도록 했습니다.

1. **자세 분석**

자세를 바탕으로 일별, 주간, 지속시간 관련 통계를 제공하여 사용자가 자세습관을 인지하고 개선할 수 있도록 하고자 했습니다.

### 🔸설계 및 구성도

![image](https://github.com/ji1won/Graduation_Project/assets/141638383/765682d0-c03a-4f09-9194-bad3dfd1a3bc)

### 🔸모델 개발

1. **데이터셋 만들기**

![image](https://github.com/ji1won/Graduation_Project/assets/141638383/a34874c8-fc37-4ad0-9112-5e4cc2ad46b2)

chin : 턱을 괸 채로 있는 자세
head down : 고개를 너무 숙인 채로 공부하는 자세
Right pose : 올바른 자세
Tilted : 고개가 좌/우로 너무 치우쳐진 자세
Turtleneck : 목이 너무 앞으로 나온 거북목 자세

1. **자세별 랜드마크 시각화**

![image](https://github.com/ji1won/Graduation_Project/assets/141638383/fb39954e-ed3d-42b0-8c5d-a6459d8a76dc)

1. **모델 훈련**

키포인트들의 numpy값들을 이용해 데이터를 준비하고 테스트 데이터를 전체 데이터의 0.05로 지정함. Epoch : 300으로 설정 후 DNN 모델로 훈련 진행

1. **훈련 결과**

![image](https://github.com/ji1won/Graduation_Project/assets/141638383/6f3eb43e-17fc-4762-9b0f-8dcc3194fd63)

![image](https://github.com/ji1won/Graduation_Project/assets/141638383/915d3409-49a7-43c0-a24f-bde3d378cd63)

### 🔸개발 환경

![image](https://github.com/ji1won/Graduation_Project/assets/141638383/b97785e7-22fa-4104-a7fe-8e2db3db7c20)

### 🔸최종 구성 웹 화면

![image](https://github.com/ji1won/Graduation_Project/assets/141638383/6d8c1201-cb64-4161-be67-1323db9dc1e4)
