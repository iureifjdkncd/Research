## Research Archive 

### 1.) Recipe Based Anomaly Detection with Adaptable Learning(2025)
#### 📌 연구 목적  
사출 성형 제조 공정에서 개별 제품 수준의 라벨링이 부족한 현실을 고려하여,  
**Recipe(Setting변동 기반) 단위의 이상 탐지 모델**을 제안.  
불량 데이터가 부족 상황에서도 신뢰성 있는 품질 예측 가능성을 확보하는 데에 초점.

---

#### 🧠 핵심 방법  
- 사출 생산품의 Setting값 기반으로 **K-Means 클러스터링** 수행 → 'Recipe' 단위 분류  
- 각 Recipe에 대해 **Basic AutoEncoder 기반 이상탐지 모델** 학습  
- 새로운 Setting기반 Recipe 생산 시, **KL Divergence** 기반 유사 Recipe 선택 → **적응형 추론** 수행 (재학습 없이 예측)

---

#### ✅ 주요 성과  
- 제안 모델은 기존 통합 품목형 학습 기반 이상탐지 모델 대비,  
  놓쳤던 불량까지 포착하여 **불량탐지 정확도 약 30 ~ 50% 향상**   
- **라벨링 문제점이 발생할 수 있는 제조 데이터 환경에서의 실질적 활용 가능성** 검증

---

#### 💡 연구 의의  
- **생산 Setting 조건 변화에 강건하고 재학습 부담이 낮은** 구조 제안  
- **지속가능한 스마트팩토리 예지보전 시스템 구축**에 기여할 수 있는 기술 제시

---

#### 📄 논문 정보  
**Lee, J., Jang, J., Tang, Q., & Jung, H.** (2025)  
*Recipe Based Anomaly Detection with Adaptable Learning: Implications on Sustainable Smart Manufacturing*  
🔗 [논문 바로가기 (MDPI Sensors)](https://doi.org/10.3390/s25051457)

---

#### 📝 프로젝트 연계  
- [비지도 학습 기반 사출 품질 예측 및 최적 세팅 추천 시스템](https://github.com/iureifjdkncd/B2B_AI_Projects/tree/main/Project_A) 에 아이디어 적용 


--- 
### A Two-Stage Deep Learning Framework for Uncertainty-Aware Forecasting and Conditional Process Optimization Toward Sustainable Smart Manufacturing (2025)

#### 📌 연구 목적
스마트 제조 환경에서 **예측 정확도**와 **불확실성 정량화**를 동시에 달성하고, 목표 품질 조건에 맞춰 **현실성 있는 공정 입력 시퀀스**를 생성·최적화할 수 있는 **2단계 딥러닝 프레임워크**를 제안.

- 제조 공정의 **변동성**, **데이터 노이즈**, **조건 변화**에 강건하게 대응  
- **실시간 시뮬레이션**과 **적응형 공정 제어**를 통한 **자원 효율성** 및 **지속가능성** 확보  

---

#### 🧠 핵심 방법

#### 1. **2단계 구조**

##### (1) 불확실성 기반 예측 모듈
- **DeepAR** 및 **Monte Carlo Dropout** 기반 시계열 예측
- **점추정** + **예측 구간(신뢰구간)** 제공
- 성능 평가:  
  - **Balanced-MAE** → 정확도 + 변동성 반영  
  - **Unified Uncertainty Score** → Prediction Interval의 Coverage + Sharpness 종합 반영  

##### (2) 조건부 생성 모듈 (LSTM-CVAE)
- 목표 품질값(Condition Vector)에 대응하는 다변량 시계열 입력 시퀀스 생성
- **Augmented Condition Vector**: 최근 입력 통계(평균, 표준편차)를 Condition Vector에 포함해 **추세·변동성** 반영
- **KL Annealing + Target-Aware Penalty**: Peak Value 및 이상 근사 Target에 대한 시퀀스 학습 강화
- Encoder·Decoder 단계모두 **Posthoc Gaussian Noise** 삽입 → Reconstruction 일반화 및 성능 향상

---

#### 2. **2단계 필터링 (Two-Stage Filtering)**

1. **재구성 오류 기반 1차 필터링**  
   - 훈련 데이터에서 관찰된 **최대 MAE** 초과 시퀀스 제거
2. **목표 근접도 기반 2차 선택**  
   - 남은 후보 중 **시계열 예측값이 타겟과 가장 가까운 시퀀스** 최종 선택 (**불확실성 기반 예측 모듈 기반**)

---

#### 3. **강건성 확보 메커니즘**
- **Batch-wise Temporal Shuffling)** 환경에서도 안정적 성능 유지
- **국소 시계열 패턴(Local Temporal Pattern)** 중심 학습 → 생산 스케줄 변화에도 대응 가능

---

#### ✅ 주요 성과
- 실제 현장 제조 데이터+ 공개 벤치마크 데이터에서 **Forecast-Aware Reconstruction** 시스템 적용 시
  - **Balanced-MAE 최대 35.7% 개선**  
  - **불확실성 점수 최대 25.2% 개선**
- 재구성된 입력 시퀀스 기반 Target품질 예측 성능 ≈ 실제 테스트 데이터 예측 성능 → **목표 기반 조건 생성 신뢰성 검증**
- 극단적인 타겟(최소/최대값) 조건에서도 **현실성 있는 시퀀스 생성 성공**
- **시뮬레이션·예측·최적화**가 통합된 제조 공정 제어 가능성 제시

---

#### 💡 연구 의의
- **Forecast-Aware Reconstruction Framework**  
  → 단순 예측 정확도뿐 아니라 **목표 일관성**과 **현실적 공정 경로** 제공
- **적응형 시뮬레이션** 기반 실시간 공정 제어·자원 최적화  
  → 에너지 절감, 폐기물 최소화 기여
- 제조 환경의 **변동성·불확실성·조건 변화**에 강건한 **지능형 의사결정 지원**
- 다양한 산업 도메인으로 확장 가능 → **지속가능한 스마트팩토리** 구현 기여

---

#### 📄 논문 정보  

- submit 완료 & publish 준비중


