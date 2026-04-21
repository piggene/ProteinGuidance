# Training-Free Iterative Guidance on SE(3)^N Flow Matching for Protein Motif Scaffolding

## 1. Problem Setup

### Motif Scaffolding

단백질의 기능적 핵심 부위(motif)의 3D 위치와 방향이 주어졌을 때, 이를 정확히 포함하면서 물리적으로 유효한 전체 단백질 backbone(scaffold)을 생성하는 문제.

- 입력: motif residue 집합 $M$, 각 residue $i \in M$의 target frame $T_i^* = (R_i^*, x_i^*) \in SE(3)$
- 출력: 전체 backbone $T = [T^1, \ldots, T^N] \in SE(3)^N$, where $T^i \approx T_i^*$ for $i \in M$
- 성공 기준: motif RMSD < 1 Å, scRMSD < 2 Å

### Base Model: FoldFlow++

SE(3)^N flow matching 기반 unconditional protein backbone 생성 모델.

- 단백질 $N$개 residue = $T = [T^1, \ldots, T^N] \in SE(3)^N$
- $T^i = (R^i \in SO(3),\ x^i \in \mathbb{R}^3)$: $i$번째 잔기의 backbone frame
- Flow: noise prior ($t=1$, $R \sim \text{Uniform}(SO(3))$, $x \sim \mathcal{N}(0, \sigma^2 I)$) 에서 data ($t=0$) 로의 geodesic path
- Velocity field: IPA transformer가 전체 backbone context를 보고 각 residue의 velocity 예측
- 모델 출력: $v_\theta^R(T_t, t)$ (rotation vectorfield), $v_\theta^x(T_t, t)$ (translation vectorfield), $\hat{T}_1$ (clean data 점 추정)

---

## 2. Method Overview

기존 접근법(ReFT 등)은 motif constraint를 만족시키기 위해 모델을 재학습하거나 hard mask를 적용한다. 우리 방법은 **모델 가중치를 전혀 수정하지 않고**, test-time에 energy function $J$를 통한 guidance만으로 motif scaffolding을 수행한다.

핵심 구성요소 3가지:

1. **Sim-MC Guidance**: gradient-free importance-weighted velocity correction
2. **Hybrid Prior Backtracking**: 모델 예측을 활용한 후보 생성 (curse of dimensionality 해결)
3. **Iterative Re-noising Refinement**: SDEdit-style 반복 정제로 점진적 수렴

---

## 3. Energy Function $J$

Motif scaffolding을 위한 cost function:

$$J(T) = \alpha \sum_{i \in M} \| \log(R_i^{*\top} R_i) \|_F^2 + \beta \sum_{i \in M} \| x_i - x_i^* \|^2$$

- 첫째 항: SO(3) geodesic distance² (rotation 정확도)
- 둘째 항: Euclidean distance² (position 정확도)
- $J$는 non-differentiable해도 무방 (gradient-free method)

---

## 4. Sim-MC Guidance (Single Step)

매 reverse step $t$에서 base velocity에 guidance correction $g_t$를 더한다:

$$\frac{dT_t}{dt} = T_t \cdot \left( v_\theta(T_t, t) + \lambda \cdot g_t(T_t, t) \right)$$

$g_t$ 계산 과정 (한 reverse step 내에서):

### Step 1: Implied Prior Backtracking

모델의 clean data 예측 $\hat{T}_1$과 현재 상태 $T_t$로부터, 이 geodesic의 출발점이었을 noise prior $\hat{T}_0$을 역산한다.

Geodesic 관계 $T_t = T_0 \cdot \exp(t \cdot \log(T_0^{-1} T_1))$ 를 역으로 풀면:

$$\hat{R}_0 = R_t \cdot \exp\!\left( \frac{-t}{1-t} \cdot \log(R_t^\top \hat{R}_1) \right)$$

$$\hat{x}_0 = \frac{x_t - t \cdot \hat{x}_1}{1 - t}$$

### Step 2: Prior Perturbation → K Candidates

$\hat{T}_0$ 주위에 small noise로 $K$개 후보를 생성한다:

$$R_0^{(k)} = \hat{R}_0 \cdot \exp(\text{hat}(\sigma_R \cdot t \cdot \epsilon_R^{(k)})), \quad \epsilon_R^{(k)} \sim \mathcal{N}(0, I)$$

$$x_0^{(k)} = \hat{x}_0 + \sigma_x \cdot t \cdot \epsilon_x^{(k)}, \quad \epsilon_x^{(k)} \sim \mathcal{N}(0, I)$$

$\sigma$에 $t$를 곱하는 이유: geodesic extrapolation이 perturbation을 $(1-t)/t$ 배 증폭하므로, $\sigma \cdot t$로 상쇄하여 data 쪽 effective spread를 일정하게 유지.

### Step 3: Geodesic Extrapolation

각 $T_0^{(k)}$에서 $T_t$를 경유하여 clean data endpoint $T_1^{(k)}$를 외삽한다:

$$R_1^{(k)} = R_0^{(k)} \cdot \exp\!\left( \frac{1}{t} \cdot \log(R_0^{(k)\top} R_t) \right)$$

$$x_1^{(k)} = x_0^{(k)} + \frac{x_t - x_0^{(k)}}{t}$$

이것이 원본 GGPS의 핵심 공식. "만약 noise가 $T_0^{(k)}$였다면, 현재 $T_t$를 경유하는 geodesic의 끝점은 $T_1^{(k)}$"

### Step 4: Importance Weighting

각 후보에 $J$를 평가하고 importance weight를 계산:

$$\alpha_k = \frac{\exp(-J(T_1^{(k)}) / \tau)}{\frac{1}{K}\sum_j \exp(-J(T_1^{(j)}) / \tau)} - 1$$

$J$가 낮은(motif에 가까운) 후보에 높은 weight, 높은 후보에 음의 weight.

### Step 5: Weighted Velocity Aggregation

각 후보 방향의 velocity를 가중 평균:

$$g_t = \frac{1}{K} \sum_{k=1}^{K} \alpha_k \cdot v(T_t \to T_1^{(k)})$$

여기서 $v(T_t \to T_1^{(k)})$는 FoldFlow++의 vectorfield 단위로 변환된 conditional velocity:

$$v_R = R_t \cdot \log(R_1^{(k)\top} R_t) \cdot s_\text{rot}, \qquad v_x = s_\text{coord} \cdot \frac{x_t - x_1^{(k)}}{t}$$

---

## 5. Iterative Re-noising Refinement

단일 reverse pass로는 motif RMSD < 1 Å 도달이 어렵다 (실험: ~6-9 Å). 이를 해결하기 위해 SDEdit-style iterative refinement을 적용한다.

### 알고리즘

```
Input:  base model v_θ, energy J, schedule [(t_1, λ_1, n_1), ..., (t_R, λ_R, n_R)]
Output: refined protein T*

T_clean ← None
for r = 1, ..., R:
    if r == 1:
        T_init ← sample_noise_prior()                    // t = 1.0
    else:
        T_init ← re_noise(T_clean, t_start=t_r)          // 이전 결과를 t_r까지 re-noise

    T_clean ← guided_reverse(T_init, t_start=t_r, λ=λ_r, num_steps=n_r)

return T_clean
```

### Re-noising 연산

이전 round의 clean output $T_\text{clean}$을 time $t$까지 re-noise:

$$R_t = R_\text{clean} \cdot \exp\!\left( t \cdot \log(R_\text{clean}^\top R_\text{noise}) \right), \quad R_\text{noise} \sim \text{Uniform}(SO(3))$$

$$x_t = (1-t) \cdot x_\text{clean} + t \cdot x_\text{noise}, \quad x_\text{noise} \sim \mathcal{N}(0, I)$$

### Schedule 설계 원칙

| Round | $t_\text{start}$ | 의미 |
|-------|-----------------|------|
| 1 | 1.0 | 전체 noise → 대략적 형태 생성 |
| 2 | 0.5 | 50% 보존 → 중간 스케일 교정 |
| 3 | 0.3 | 70% 보존 → 세부 교정 |
| 4-5 | 0.2 → 0.1 | 80-90% 보존 → fine-tuning |
| 6-8 | 0.07 → 0.03 | 93-97% 보존 → 최종 polish |

$\lambda$는 전 round 동일(=5.0)이 가장 안정적. $\lambda$를 round마다 증가시키면 flow가 깨져서 발산함을 실험적으로 확인.

---

## 6. 기존 방법과의 비교

| | ReFT | Fixed Mask (Inpainting) | Ours |
|---|---|---|---|
| 모델 수정 | O (weight update) | X | X |
| Task별 비용 | 재학습 필요 | 없음 | 없음 |
| J 조건 | differentiable 필요 | 해당 없음 | **gradient-free** (임의 J) |
| Motif 처리 | learned | hard constraint | soft guidance |
| 새 motif 적용 | 재학습 | 바로 | J 교체 후 바로 |
| 성능 (motif RMSD) | < 1 Å | ≈ 0 Å | 1.3 - 1.8 Å |

핵심 차별점:

1. **Training-free**: 모델 가중치를 건드리지 않음. 같은 base model로 아무 task에 적용 가능.
2. **Gradient-free**: $J$가 미분 불가능해도 동작. ESMFold, ProteinMPNN 등 black-box evaluator를 $J$로 직접 사용 가능.
3. **SE(3)^N product space 최초 적용**: 기존 GGPS는 single SE(3)에서만 검증. 본 연구에서 SE(3)^N으로 확장하며 curse of dimensionality를 hybrid prior backtracking으로 해결.

---

## 7. Experimental Results

### Setup

- Base model: FoldFlow++ (FF2, unconditional)
- Protein length: 60 residues
- Motif: 7 residues (indices 20-26), synthetic target
- Hyperparameters: $K=16$, $\lambda=5.0$, $\sigma_R=0.5$, $\sigma_x=1.0$
- 8 rounds, schedule: $t_\text{start} \in [1.0, 0.5, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03]$

### Results

| Method | Motif Cα-RMSD |
|--------|--------------|
| Baseline (no guidance) | 11.89 Å |
| Single-round guidance | 9.27 Å |
| **Iterative 8-round (single sample)** | **1.30 Å** |
| **Iterative 8-round (best-of-8)** | **1.46 Å** |
| **Iterative 8-round (mean ± std, N=8)** | **1.83 ± 0.34 Å** |

### Success Rate (N=8 samples)

| 기준 | 달성 비율 |
|------|----------|
| < 5 Å | 8/8 (100%) |
| < 3 Å | 8/8 (100%) |
| < 2 Å | 6/8 (75%) |
| < 1 Å | 0/8 (0%) |

### Round별 수렴

```
Round:  1     2     3     4     5     6     7     8
RMSD:  9.3 → 5.3 → 5.0 → 3.9 → 2.4 → 1.7 → 1.5 → 1.3 Å
```

### Ablation: λ Schedule 비교

| Schedule | Final RMSD | 비고 |
|----------|-----------|------|
| Conservative (λ=5 고정) | **1.30 Å** | 안정적 수렴 |
| Aggressive (λ 증가) | 4.41 Å | Round 3에서 불안정 |
| Lambda ramp (λ=2→30) | 18.56 Å | 완전 발산 |

---

## 8. Limitations & Future Work

### 현재 한계

1. **1 Å 미만 미도달**: strict motif scaffolding 기준(< 1 Å) 충족 못 함. 1.30 Å로 근접했으나 gap 존재.
2. **Designability 미검증**: scRMSD, pLDDT 등 단백질 품질 메트릭 미측정. ESMFold/ProteinMPNN pipeline 연동 필요.
3. **Synthetic motif만 테스트**: Watson et al. 2023의 24개 실제 PDB benchmark에서의 성능 미확인.
4. **Scaffold 구조 품질**: backbone이 물리적으로 타당한지 (bond length, Ramachandran 등) 미검증.

### 향후 방향

1. **Watson benchmark 24-problem 평가**: 실제 PDB motif으로 검증
2. **J 확장**: motif RMSD 외에 clash penalty, secondary structure, radius of gyration 등 추가
3. **Fixed mask + guidance 조합**: motif는 hard fix, scaffold 품질은 guidance로 최적화
4. **K 증가 및 schedule 최적화**: K=64+, 더 많은 round, adaptive λ
5. **Non-differentiable J 활용**: ESMFold confidence, ProteinMPNN perplexity를 J로 직접 사용
