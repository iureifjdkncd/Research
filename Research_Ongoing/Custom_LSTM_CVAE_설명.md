# Augmented Condition Vector & LSTM-CVAE

## 1) Augmented Condition Vector

### 📌 정의
기존 Condition Vector(목표값 **c<sub>t</sub>**)에 **최근 다변량 입력 시퀀스의 통계적 특징(평균, 표준편차)**을 추가하여  
Generative 모델이 입력시계열의 **추세(Trend)**와 **변동성(Volatility)**을 함께 인식하도록 강화한 구조.

---

### 🔹 생성 과정
최근 N′개의 입력시퀀스  

\[
\hat{x}_{t-N′:t-1} \in \mathbb{R}^{N′ \times T \times F}
\]

에서 각 Feature별 통계치를 계산(Target변수 제외):

\[
\mu_t = \frac{1}{N′} \sum_{i=t-N′}^{t-1} \hat{x}^i
\]
\[
\sigma_t = \frac{1}{N′} \sum_{i=t-N′}^{t-1} (\hat{x}^i - \mu_t)^2
\]

- **μ<sub>t</sub>**: 각 Feature의 평균 → **Trend 반영**  
- **σ<sub>t</sub>**: 각 Feature의 표준편차 → **Volatility 반영**

> N′ = 0 (이전 데이터 없음) → μ<sub>t</sub> = 0, σ<sub>t</sub> = 0

---

### 🔹 Condition Vector 추가용 Scalar 변환
모든 Feature에 대한 평균값을 구해 스칼라 형태로 축약:

\[
\bar{\mu}_t = \frac{1}{F} \sum_{j=1}^F [\mu_t]_j
\]
\[
\bar{\sigma}_t = \frac{1}{F} \sum_{j=1}^F [\sigma_t]_j
\]

---

### 🔹 최종 Augmented Condition Vector 생성성
원래의 조건 벡터 **c<sub>t</sub>**와 위 두 스칼라를 연결:

\[
\tilde{c}_t = \text{concat}(c_t, \bar{\mu}_t, \bar{\sigma}_t)
\]
- 결과 차원: **D + 2**  
- 최근 시계열 흐름 + 목표값을 함께 반영

---

## 2) LSTM-CVAE 과정 및 수식

LSTM-CVAE는 **Conditional Variational AutoEncoder(CVAE)**에 LSTM 구조를 결합하여,  
입력시계열 특성(X)과 조건 정보(Y)를 함께 인코딩·디코딩하도록 설계됨.

---

### 2.1 Encoder (조건부 LSTM 인코더)

#### **입력 정보 확장**
각 시점 입력 **x<sub>t</sub>**에 동일한 **\(\tilde{c}_t\)**를 붙여 조건 포함:
\[
x'_t = [x_t; \tilde{c}_t]
\]
- 차원: \(\mathbb{R}^{T \times (F + D_c)}\),  
  \(D_c = D + 2\)

---

#### **Gaussian Noise 추가 (강건성 확보)**
\[
\tilde{c}_t \leftarrow \tilde{c}_t + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2 I)
\]

---

#### **LSTM 인코딩**
\[
h_{\text{enc}} = \text{LSTM}(x'_{1:T})
\]
최종 히든 스테이트 **h<sub>enc</sub>**가 전체 시퀀스와 조건 요약.

---

#### **Latent Vector 분포 추정**
\[
\mu = W_\mu h_{\text{enc}} + b_\mu
\]
\[
\log\sigma^2 = W_{\log\text{var}} h_{\text{enc}} + b_{\log\text{var}}
\]

---

### 2.2 Latent Sampling (Reparameterization Trick)
\[
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]
- 학습 중 **역전파 가능**하도록 재파라미터화

---

### 2.3 Decoder (조건부 LSTM 디코더)

#### **잠재벡터 투영**
\[
h_z = \phi(W_z z + b_z)
\]
시퀀스 길이 **T**만큼 반복(repeat)

---

#### **조건 결합**
\[
h_t^{(0)} = [h_{z,t}; \tilde{c}_t], \quad t = 1, \dots, T
\]

---

#### **LSTM 복원**
\[
h_{\text{dec}} = \text{LSTM}(h_{1:T}^{(0)})
\]

---

#### **출력 변환**
\[
\hat{x}_t = W_{\text{out}} h_{\text{dec},t} + b_{\text{out}}
\]

---

#### **Post-hoc Gaussian Noise 추가**
\[
\hat{x}_{1:T} \leftarrow \hat{x}_{1:T} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2 I)
\]

---

### 2.4 Loss Function (ELBO 기반)

#### **Reconstruction Loss (MAE)**
\[
L_{\text{recon}} = \frac{1}{T} \sum_{t=1}^T \| x_t - \hat{x}_t \|_1
\]

#### **KL Divergence **
\[
L_{\text{KL}} = D_{\text{KL}}\big(q(z|x_t, \tilde{c}_t) \parallel p(z|\tilde{c}_t)\big)
\]

---

#### **KL Annealing**
\[
L_{\text{total}} = L_{\text{recon}} + \lambda_{\text{KL}} L_{\text{KL}}, \quad \lambda_{\text{KL}} \in [0,1]
\]

---

#### **Target-Aware Penalty (고타겟 가중치)**
\[
L_{\text{weighted}} = \big(1 + I(c_t \ge \tau) \cdot p\big) L_{\text{total}}
\]
- Peak 또는 이상근접 타겟의 재구성 품질 강화

---

## 📌 정리
- **Augmented Condition Vector**  
  → 목표값(Y) + 최근 입력 시계열(X)의 통계 특징(평균·표준편차) 제공 → **시계열 패턴 반영력 강화**
- **LSTM-CVAE**  
  → Condition Vector를 인코더·디코더 모두에 삽입  
  → KL 정규화, 노이즈, 가중 손실로 **현실성·변동성 유지 시퀀스 생성**
