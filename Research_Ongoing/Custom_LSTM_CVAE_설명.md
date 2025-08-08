# Augmented Condition Vector기반 Custom LSTM-CVAE

## 1) Augmented Condition Vector

### 📌 정의
기존 Condition Vector(목표값 $c_t$)에 **최근 다변량 입력 시퀀스의 통계적 특징(평균, 표준편차)**을 추가하여  
Generative 모델이 입력 시계열의 **추세(Trend)**와 **변동성(Volatility)**을 함께 인식하도록 강화한 구조.

---

### 🔹 생성 과정
최근 $N'$개의 입력 시퀀스

$$
\hat{x}_{t-N':t-1} \in \mathbb{R}^{N' \times T \times F}
$$

각 Feature별 통계치 계산(Target 변수 제외):

$$
\mu_t = \frac{1}{N'} \sum_{i=t-N'}^{t-1} \hat{x}^i
$$

$$
\sigma_t = \frac{1}{N'} \sum_{i=t-N'}^{t-1} (\hat{x}^i - \mu_t)^2
$$

- $\mu_t$: 각 Feature의 평균 → **Trend 반영**  
- $\sigma_t$: 각 Feature의 표준편차 → **Volatility 반영**

> $N' = 0$ (이전 데이터 없음) → $\mu_t = 0$, $\sigma_t = 0$

---

### 🔹 Condition Vector 추가용 Scalar 변환
모든 Feature에 대한 평균값을 구해 스칼라 형태로 축약:

<p align="center">
  <img width="200" height="50" alt="화면 캡처 2025-08-08 103606"
       src="https://github.com/user-attachments/assets/8563cd1f-cb18-4a97-9783-9e5514c24b14" />
</p>


---

### 🔹 최종 Augmented Condition Vector 생성
원래의 조건 벡터 $c_t$와 위 두 스칼라를 연결:

$$
\tilde{c}_t = \mathrm{concat}(c_t, \bar{\mu}_t, \bar{\sigma}_t)
$$

- 결과 차원: $D + 2$  
- 최근 시계열 흐름 + 목표값을 함께 반영

---

## 2) LSTM-CVAE 과정 및 수식

LSTM-CVAE는 **Conditional Variational AutoEncoder(CVAE)**에 LSTM 구조를 결합하여,  
입력 시계열 특성($X$)과 조건 정보($Y$)를 함께 인코딩·디코딩하도록 설계됨.

---

### 2.1 Encoder (조건부 LSTM 인코더)

#### **입력 정보 확장**
각 시점 입력 $x_t$에 동일한 $\tilde{c}_t$를 붙여 조건 포함:

$$
x'_t = [x_t; \tilde{c}_t]
$$

- 차원: $\mathbb{R}^{T \times (F + D_c)}$, $D_c = D + 2$

---

#### **Gaussian Noise 추가 (강건성 확보)**
<p align="center">
<img width="200" height="45" alt="화면 캡처 2025-08-08 103912" src="https://github.com/user-attachments/assets/9156c24b-4f07-49eb-8d18-a26f219999c3" />
</p>

---

#### **LSTM 인코딩**
$$
h_{\mathrm{enc}} = \mathrm{LSTM}(x'_{1:T})
$$

---

#### **Latent Vector 분포 추정**
$$
\mu = W_\mu h_{\mathrm{enc}} + b_\mu
$$

$$
\log \sigma^2 = W_{\log\mathrm{var}} h_{\mathrm{enc}} + b_{\log\mathrm{var}}
$$

---

### 2.2 Latent Sampling (Reparameterization Trick)
$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

---

### 2.3 Decoder (조건부 LSTM 디코더)

#### **Latent Vector 투영**
<p align="center">
<img width="100" height="70" alt="화면 캡처 2025-08-08 104058" src="https://github.com/user-attachments/assets/011f09ea-79a5-4c99-9844-b90874ad7dca" />
</p>
(시퀀스 길이 $T$만큼 반복)

---

#### **Condition 결합**
$$
h_t^{(0)} = [h_{z,t}; \tilde{c}_t], \quad t = 1, \dots, T
$$

---

#### **LSTM 복원**
$$
h_{\mathrm{dec}} = \mathrm{LSTM}(h_{1:T}^{(0)})
$$

---

#### **출력 변환**
<p align="center">
<img width="210" height="30" alt="화면 캡처 2025-08-08 104315" src="https://github.com/user-attachments/assets/d6937408-85fd-477a-b807-4810d397667c" />
</p>

---

#### **Post-hoc Gaussian Noise 추가**
<p align="center">
<img width="210" height="30" alt="화면 캡처 2025-08-08 104436" src="https://github.com/user-attachments/assets/96a9e602-1b72-42ee-b8b0-e0764b997078" />
</p>

---

### 2.4 Loss Function (ELBO 기반)

#### **Reconstruction Loss (MAE)**
$$
L_{\mathrm{recon}} = \frac{1}{T} \sum_{t=1}^T \| x_t - \hat{x}_t \|_1
$$

#### **KL Divergence**
$$
L_{\mathrm{KL}} = D_{\mathrm{KL}}\big(q(z|x_t, \tilde{c}_t) \,\|\, p(z|\tilde{c}_t)\big)
$$

#### **KL Annealing**
$$
L_{\mathrm{total}} = L_{\mathrm{recon}} + \lambda_{\mathrm{KL}} L_{\mathrm{KL}}, \quad \lambda_{\mathrm{KL}} \in [0,1]
$$

#### **Target-Aware Penalty**
$$
L_{\mathrm{weighted}} = \big(1 + I(c_t \ge \tau) \cdot p\big) \, L_{\mathrm{total}}
$$

---

## 📌 정리
- **Augmented Condition Vector**  
  → 목표값($Y$) + 최근 입력 시계열($X$)의 통계 특징(평균·표준편차) 제공 → **시계열 패턴 반영력 강화**
- **LSTM-CVAE**  
  → Condition Vector를 인코더·디코더 모두에 삽입  
  → KL 정규화, 노이즈, 가중 손실로 **현실성·변동성 유지 시퀀스 생성**

