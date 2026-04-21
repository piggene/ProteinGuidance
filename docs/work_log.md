# Work Log: Training-Free Iterative Guidance on FoldFlow++

## 프로젝트 개요

FoldFlow++ (SE(3)^N flow matching for protein backbone generation) 위에 GGPS (Guided Grasp Pose Sampler)의 training-free test-time guidance를 적용하여 motif scaffolding을 수행.

**핵심 contribution**: SE(3)^N (Lie group product space) 위에서 최초의 training-free flow matching guidance framework

---

## 환경 설정

```bash
conda activate foldflow          # Python 3.10, torch 2.4.1+cu121
cd /PublicSSD/genechung/FoldFlow
# FF2 체크포인트: ckpt/ff2_base.pth (GitHub Release 0.2.0에서 다운로드)
# GPU: RTX 3090 x2, gpu_id=1 사용 (공유 서버)
```

---

## 코드 구조

### 새로 만든 파일들

```
foldflow/guidance/
├── __init__.py              # 모듈 export
├── se3n_utils.py            # SE(3)^N 기하 primitives
│   ├── so3_log_skew()       # SO(3) → so(3) log map (배치 지원)
│   ├── skew_from_vec()      # hat operator
│   ├── so3_exp_skew()       # so(3) → SO(3) exp map
│   ├── velocity_toward()    # x_t에서 x_1으로의 velocity (FF2 단위 매칭)
│   ├── perturb_frames()     # K개 가우시안 섭동 후보 생성
│   ├── geodesic_backtrack() # (x_t, x_1) → implied x_0 역산
│   ├── geodesic_extrapolate() # (x_0, x_t) → x_1 외삽
│   ├── sample_prior()       # Uniform(SO3) × N(0,I) 샘플링
│   ├── renoise_frames()     # SDEdit-style re-noising
│   └── rigids_to_rot_trans() # [B,N,7] → (R,x)
├── energies.py              # J_motif: SO(3) geodesic² + Euclidean²
├── sim_mc.py                # SimMCGuidance (핵심 guidance class)
├── mc.py                    # MCGuidance (reference bank 방식, 미완성)
├── guided_inference.py      # reverse loop + guidance hook (t_start 지원)
└── guided_sampler.py        # 모델 로드 + sample() API (init_rigids 지원)

scripts/
├── test_baseline.py         # 무guidance sanity check
├── test_guidance.py         # planted mask + guidance 테스트
├── run_and_visualize.py     # single-round sweep 실험 + 시각화
├── run_iterative.py         # iterative refinement 비교 실험
├── run_full_eval.py         # 종합 평가 (extended + best-of-N + PDB)
└── run_push_limit.py        # sub-1Å 도전 실험 (아직 미실행)

docs/
├── method.md                # 방법론 정리 (노션 복붙용)
└── work_log.md              # 이 파일

results_guidance/             # 실험 결과
├── *.png                    # 시각화
├── *.pdb                    # 단백질 구조 파일
└── results.npz              # 수치 결과
```

### FoldFlow++ 기존 코드에서 중요한 위치

- `runner/train.py:931-951` — Experiment.inference_fn (reverse loop 원본)
- `foldflow/models/so3_fm.py:51` — SO3FM.sample_ref (noise prior)
- `foldflow/models/r3_fm.py:47` — R3FM.sample_ref
- `foldflow/models/se3_fm.py:192` — SE3FM.sample_ref
- `foldflow/utils/so3_condflowmatcher.py:26` — sample_xt (geodesic interpolation)

### 핵심 convention (실수하기 쉬운 것들)

- **t=1이 noise, t=0이 data** (reverse는 1→0)
- **fixed_mask=1이 motif (frozen)**, flow_mask = (1-fixed_mask)*res_mask
- **Vectorfield 단위 매칭 필수**:
  - rot_vf = R_t @ log(R_data^T R_t) × so3_inference_scaling (default 10.0)
  - trans_vf = coordinate_scaling × (x_t - x_data) / (t + eps) (coordinate_scaling default 0.1)
  - 이거 안 맞으면 anti-guidance 됨

---

## 실험 히스토리 및 결과

### 1차: 순수 perturbation 방식 (data-side gaussian)

모델 예측 rigid_pred 주위에 σ로 K개 후보를 직접 흩뿌림.

```python
R_k = R_pred @ exp(hat(σ_rot · ε))
x_k = x_pred + σ_trans · ε
```

결과 (single-round, length=60, motif 7개):
- Baseline: 10.84 Å
- Best (K16 λ=2 σ_trans=10): 7.79 Å (ratio 0.719, 28% 감소)
- λ 너무 크면 오히려 나빠짐 (K16 λ=5: ratio 1.674)

### 2차: 원본 GGPS 방식 (random prior + geodesic extrapolation)

random x_0 ~ prior에서 geodesic extrapolation으로 x_1 후보 생성. σ 없음.

```python
x_1^(k) = x_0^(k) · exp(log(x_0^{k,-1} x_t) / t)
```

결과: **완전 실패** (RMSD 241~56000 Å). Curse of dimensionality — SE(3)^60 (360차원)에서 K=32로는 공간 커버 불가.

### 3차: Hybrid (backtrack + perturb + extrapolate)

모델 예측으로 implied x_0를 역산하고, 그 주위에서 perturbation → geodesic extrapolation.

```python
x_0_impl = geodesic_backtrack(x_t, rigid_pred, t)
x_0^(k) = perturb(x_0_impl, σ · t)  # σ에 t 곱해서 extrapolation 증폭 상쇄
x_1^(k) = geodesic_extrapolate(x_0^(k), x_t, t)
```

결과 (single-round):
- Best (K16 λ=5 σ=0.5): 6.58 Å (ratio 0.607, 39% 감소)
- σ=0.5 > σ=1.0, λ=5 sweet spot, K=16이면 충분

### 4차: Iterative re-noising refinement

SDEdit-style로 매 round 이전 결과를 조금만 re-noise한 후 다시 guided denoise.

```
Round 1: t=1.0→0, 50 steps → 9.3 Å
Round 2: re-noise to 0.5, 30 steps → 5.3 Å
...
Round 8: re-noise to 0.03, 6 steps → 1.3 Å
```

**Best single sample (seed=42): 1.30 Å**

Schedule 비교:
- **Conservative (λ=5 고정)**: 1.30 Å — 안정적 수렴
- Aggressive (λ 증가): 4.41 Å — Round 3에서 불안정
- Lambda ramp (λ=2→30): 18.56 Å — 완전 발산

**결론: λ 고정, t_start만 줄이는 게 최선.**

### 5차: Best-of-N + Extended evaluation

8-round iterative를 8개 seed로 실행:

```
Mean: 1.83 ± 0.34 Å
< 2 Å: 6/8 (75%)
< 3 Å: 8/8 (100%)
< 1 Å: 0/8 (0%)
```

### 미실행: run_push_limit.py

12-round, K=32/64, best-of-16. GPU 사용 중이라 미실행 상태.
Schedule A~D 4가지 + best-of-16. 실행 명령:

```bash
conda run -n foldflow python scripts/run_push_limit.py
```

---

## 핵심 hyperparameter 가이드

| 파라미터 | 역할 | 추천값 | 주의 |
|---------|------|--------|------|
| K | MC 후보 수 | 16-32 | 높을수록 좋지만 메모리 증가 |
| λ | guidance 강도 | 5.0 | >10이면 flow 깨짐, 고정 추천 |
| σ_rot | prior perturbation (rad) | 0.5 | 코드에서 t를 곱해서 time-scaled |
| σ_trans | prior perturbation | 1.0 | 코드에서 t를 곱해서 time-scaled |
| temperature | importance weight softmax | 1.0 | |
| num_t | reverse step 수 | 50 (round 1), 6-15 (later) | |
| t_start | re-noise 시작점 | 1.0→0.03 decreasing | |
| rounds | iterative round 수 | 8-12 | 더 많을수록 수렴 |

---

## 남은 TODO

1. **run_push_limit.py 실행**: 12-round + K=64 + best-of-16으로 sub-1Å 도전
2. **Watson et al. 24 benchmark**: 실제 PDB motif으로 평가
3. **Designability 검증**: ESMFold + ProteinMPNN pipeline으로 scRMSD, pLDDT 측정
4. **J 확장**: clash penalty, secondary structure propensity 등 추가
5. **Fixed mask + guidance 조합**: motif hard fix + scaffold 품질 guidance
6. **MCGuidance 완성**: reference bank (natural PDB backbones) 구축

---

## 관련 논문

- FoldFlow (ICLR 2024): arXiv:2310.02391
- FoldFlow++ (NeurIPS 2024): arXiv:2405.20313
- On the Guidance of Flow Matching (ICML 2025 spotlight): arXiv:2502.02150
- Reward-Guided Iterative Refinement (ICML 2025): arXiv:2502.14944
- Watson et al. motif scaffolding benchmark (2023)
