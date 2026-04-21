# Project: GGPS Guidance on FoldFlow++ (SE(3)^N Flow Matching)

## 연구 목표

FoldFlow++ (SE(3)^N flow matching for protein backbone generation) 위에
GGPS (Guided Grasp Pose Sampler)의 training-free test-time guidance를 적용.

핵심 contribution:
> **SE(3)^N (Lie group product space) 위에서 최초의 training-free flow matching guidance framework**
> 응용: Motif Scaffolding (protein design)

---

## GGPS 핵심 수학 요약

### 세팅

- Base model: flow matching on SE(3), geodesic path 가정
- x_t = x_0 · exp(κ(t) · log(x_0^{-1} x_1)), κ(t) = t (linear schedule)
- velocity field: v_θ(x_t, t) : SE(3) → se(3)
- 목표: p(x_1) · exp(-J(x_1)) 에서 샘플링 (energy-guided)

### Guided velocity field

guidance가 없는 base ODE:
```
dx_t/dt = x_t · v_θ(x_t, t)
```

guidance term g_t를 더해서:
```
dx_t/dt = x_t · (v_θ(x_t, t) + λ · g_t(x_t, t))
```

### g_t 추정 방법들

**g_cov-A (Covariance-A):**
- Euclidean approximation
- g_t ≈ -Cov[x_1|x_t]^{-1} · E[∇J(x_1)|x_t]
- SE(3)로 확장: Riemannian gradient 사용

**g_cov-G (Covariance-G):**
- Riemannian metric 기반
- g_t ≈ -G(x_t)^{-1} · ∇_{x_t} E[J(x_1)|x_t]
- G = Fisher information metric on SE(3)

**g_sim-MC (Simulation-based Monte Carlo) ← 핵심**

asymptotically exact. gradient-free.

```
g_t(x_t) ≈ -Σ_i w_i · log(x_t^{-1} x_1^{(i)}) / (1-t)

여기서:
x_1^{(i)} ~ p(x_1|x_t)  ← ODE forward로 샘플
w_i = exp(-J(x_1^{(i)})) / Σ_j exp(-J(x_1^{(j)}))  ← importance weight
```

**작동 원리:**
1. 현재 noisy state x_t에서 K개 candidate x_1^(i) 샘플링 (ODE forward)
2. 각 candidate에 J 평가
3. J 낮은 candidate에 높은 weight
4. weighted average로 guidance direction 계산
5. 이 방향으로 velocity field 수정

**장점:**
- gradient-free → J가 non-differentiable해도 OK (ESMFold, ProteinMPNN 등)
- asymptotically exact → K → ∞ 이면 정확한 posterior
- training-free → base model 수정 없이 J만 바꾸면 됨

### SE(3)에서 핵심 연산

```python
# SO(3) geodesic path
R_t = R_0 @ expm(t * logm(R_0.T @ R_1))

# SE(3) log map (at identity)
log(T) = [log(R), J^{-1} p]  # J = left Jacobian of SO(3)

# Riemannian gradient on SE(3)
grad_SE3 J(T) = T · grad_se3 J
```

---

## FoldFlow++ 구조 요약

### 단백질 표현

```
단백질 N개 residue = T = [T^1, ..., T^N] ∈ SE(3)^N
T^i = (R^i, x^i):
  R^i ∈ SO(3): i번째 잔기 backbone 방향 (N-Cα-C frame)
  x^i ∈ R³:   i번째 잔기 Cα 위치
```

### Flow 구조

- Noise prior: R^i ~ Uniform(SO(3)), x^i ~ N(0, σ²I), center of mass 제거
- Path: geodesic (t-linear schedule) — GGPS와 동일
- SE(3)^N = product space → SE(3) flow를 N번 독립 적용
- Velocity field: IPA (Invariant Point Attention) transformer
  → 전체 backbone context 보고 각 residue velocity 계산
  → residue 독립 아님! (context 공유)

### Loss

```
L = E[Σ_i (||v_θ^R(T_t,t) - v*_R^i||² + ||v_θ^x(T_t,t) - v*_x^i||²)]

v*_R^i = log(R_t^{i,T} R_1^i) / (1-t)   ← target angular velocity
v*_x^i = (x_1^i - x_t^i) / (1-t)         ← target linear velocity
```

---

## 이번에 하려는 것: Motif Scaffolding

### 문제 정의

```
M = motif residue 집합 (기능에 필수적인 잔기들)
목표: T^i = T^i_target (i ∈ M) 을 만족하면서 전체가 유효한 단백질
```

이미지 inpainting과 동일한 구조:
- 관측된 픽셀 → motif residues (고정)
- 마스킹된 픽셀 → scaffold residues (생성)

best-of-K 불가: P(motif 동시 만족) ≈ (ε³/V)^m ≈ 10^{-28}

### J 설계

```python
def J_motif(T, target_frames, motif_indices, alpha=1.0, beta=1.0, gamma=0.1):
    """
    T: [N, 4, 4] SE(3) frames
    target_frames: [m, 4, 4] target frames for motif residues
    motif_indices: list of motif residue indices
    """
    J = 0.0

    for i, idx in enumerate(motif_indices):
        R = T[idx, :3, :3]
        x = T[idx, :3, 3]
        R_target = target_frames[i, :3, :3]
        x_target = target_frames[i, :3, 3]

        # Rotation loss (SO(3) geodesic distance)
        R_diff = R_target.T @ R
        log_R = logm(R_diff)  # so(3) element
        J_rot = alpha * torch.norm(log_R, 'fro')**2

        # Translation loss
        J_pos = beta * torch.norm(x - x_target)**2

        J += J_rot + J_pos

    # Clash loss (optional)
    for i in range(N):
        if i not in motif_indices:
            for j in motif_indices:
                dist = torch.norm(T[i, :3, 3] - T[j, :3, 3])
                J += gamma * torch.clamp(3.8 - dist, min=0)**2

    return J
```

### 평가 메트릭

```
1. motif_RMSD < 1Å     (motif 위치 정확도)
2. scRMSD < 2Å         (전체 구조 designability)
   = ESMFold(ProteinMPNN(T))와 T의 Cα RMSD
3. pLDDT > 70          (ESMFold confidence)
4. Success Rate = fraction(motif_RMSD < 1Å AND scRMSD < 2Å)
```

Benchmark: Watson et al. 2023의 24개 scaffolding problem
FoldFlow++ ReFT (retraining): 24/24
GGPS (training-free): ???

---

## 코드 구조 목표

```
Guidance_MatrixLie/          ← 기존 GGPS 레포
├── CLAUDE.md                ← 이 파일
├── foldflow/                ← FoldFlow++ 레포 (submodule or copy)
│   ├── models/
│   │   └── folding_diff/
│   └── ...
├── guidance/
│   ├── base.py              ← g_t 추정 base class
│   ├── sim_mc.py            ← g_sim-MC 구현 (기존 GGPS에서 가져옴)
│   ├── cov_a.py             ← g_cov-A
│   └── cov_g.py             ← g_cov-G
├── tasks/
│   └── motif_scaffolding.py ← J 정의 + 평가
└── scripts/
    └── run_motif_guided.py  ← 실험 실행
```

---

## 작업 순서

1. FoldFlow++ 레포 세팅 + inference 확인
2. g_sim-MC를 SE(3)^N으로 확장
   - 기존: SE(3) single pose
   - 확장: SE(3)^N → J가 전체 backbone의 함수
3. J_motif 구현
4. Watson et al. 24 benchmark 실험
5. FoldFlow++ ReFT (retraining baseline) 대비 비교

---

## 관련 논문

- FoldFlow (ICLR 2024): arXiv:2310.02391
- FoldFlow++ (NeurIPS 2024): arXiv:2405.20313
- On the Guidance of Flow Matching (ICML 2025 spotlight): arXiv:2502.02150
- Reward-Guided Iterative Refinement (ICML 2025): arXiv:2502.14944
- Watson et al. motif scaffolding benchmark (2023)

---