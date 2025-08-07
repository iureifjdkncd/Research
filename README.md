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
- **제조 데이터 라벨링 부족 환경에서의 실질적 활용 가능성** 검증

---

#### 💡 연구 의의  
- **생산 Setting 조건 변화에 강건하고 재학습 부담이 낮은** 구조 제안  
- **지속가능한 스마트팩토리 구축**에 기여할 수 있는 기반 기술 제시

---

#### 📄 논문 정보  
**Lee, J., Jang, J., Tang, Q., & Jung, H.** (2025)  
*Recipe Based Anomaly Detection with Adaptable Learning: Implications on Sustainable Smart Manufacturing*  
🔗 [논문 바로가기 (MDPI Sensors)](https://doi.org/10.3390/s25051457)
