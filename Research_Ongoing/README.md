#### 내용 축약

#### 1.) 실험 데이터셋 예측모델 설정 & 성능 평가 

<img width="768" height="285" alt="화면 캡처 2025-08-08 095353" src="https://github.com/user-attachments/assets/05e2f285-0453-4771-9a41-4247e1864e51" />


#### 2.) Augmented Condition Vector 기반 제안된 LSTM-CVAE 모델 구성 

<img width="1000" height="341" alt="화면 캡처 2025-08-08 095340" src="https://github.com/user-attachments/assets/e97ea272-94a2-4512-96c3-7079393d7e22" />


#### 3.) LSTM-CVAE 기반 Reconstruction Loss 결과 & 재구성데이터의 Target 예측 성능 평가 

<img width="794" height="537" alt="화면 캡처 2025-08-08 095218" src="https://github.com/user-attachments/assets/22839308-9f19-4b85-b7b4-74d654a4947b" />


<img width="1013" height="676" alt="화면 캡처 2025-08-08 095235" src="https://github.com/user-attachments/assets/6914a128-9a1c-4cd5-9d9f-2e231a94f50f" />


#### 4.) 조건부 시계열 생성 및 최적화 예시  

- LSTM-CVAE 기반 Target 조건부 Multivariate Sequence 100개 생성

- Reconstruction MAE Distribution 기반 1차 현실성 필터링

- 2차 DeepAR / MCD CNN-LSM 등의 기존 Target 예측모델에 입력 후 현재 조건부와 최근접 추론결과 가진 Sequence 최종 선택 

<img width="983" height="476" alt="화면 캡처 2025-08-08 095322" src="https://github.com/user-attachments/assets/693937ac-38aa-4f95-babc-95107ad8b437" />
