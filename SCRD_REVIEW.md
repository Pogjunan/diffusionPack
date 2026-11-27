# SCRD 검토와 재구성 기준

검토일: 2026-11-26. 목적은 기존 SCRD의 실패 원인을 가정으로 확정하지 않고,
같은 데이터 처리와 DDPM 계산으로 모델들을 다시 비교할 수 있게 만드는 것이다.
이 문서는 논문 검토 기록이며, 사용자의 이해 완료나 논문 완독을 표시하지 않는다.

후속 연결: 기존 Prediction 진입점에 직접 붙인 경로는 [PREDICTION_SCRD.md](PREDICTION_SCRD.md)를 따른다.
해당 경로는 공통 DDPM core의 beta variance를 그대로 쓰며, 이 문서의 별도 비교 runner에서는
posterior/beta를 선택할 수 있다. 최종 통합 suite는 91개 통과, 실행별 기록은 SCRD_CHECKS.json에 있다.

## 1. 검토한 자료와 현재 판단

- SCRD 원문: *SCRD: A Spatiotemporal Cues-Guided Residual Diffusion Model for Precipitation Nowcasting*, IEEE GRSL 2024, DOI `10.1109/LGRS.2024.3486112`.
- 첨부 `SCRD(1).zip`: 현재 실행 경로는 `main/main.py`, `main/diffusion.py`, `main/module.py`, `main/data.py`다.
- 비교 기준: `Pogjunan/diffusionPack` main commit `0e819b0bc685c76c8ba510f1e4f769629b29e028`.
- 작업본은 위 main에 이미 준비한 `diffusionPack_csv_cleanup.patch`를 반영한 버전이다. 기존 CSV 정리를 새로 하지 않은 것으로 취급하지 않는다.
- 이전 단일 미래 실험과 현재 네 미래 공동 출력 구현을 구분했다. 과거 실패 checkpoint와 실제 SSI 데이터는 이번 자료에 없다.

**현재 첨부 SCRD의 기본 DDPM 수식에서 확정적인 오류를 찾지는 못했다.**
잔차 정의, forward noising, epsilon MSE, reverse mean, 마지막 단계의 noise 생략은
논문 및 기존 DDPM core와 맞는다. 합성 수치 비교의 오차는 약 `7e-8` 이하였다.
이 결과는 수식과 실행 연결을 확인한 것이며, 학습된 모델의 예측 성능을 보증하지 않는다.

반면 데이터 처리, 타깃, condition 입력, backbone, reverse variance, 학습 설정은 다르다.
Unconditional DDPM이 작은 데이터에서 잘 작동했다는 관찰만으로 conditional SCRD의
어느 모듈이 틀렸는지 결정할 수 없다. 같은 조건에서 차이를 나누어 확인해야 한다.

## 2. 논문, 첨부 코드, 재구성의 대응

페이지는 첨부 SCRD PDF의 1–5쪽을 뜻한다.

| 항목 | 논문 원문 | 첨부 SCRD 현재 구현 | 재구성 기준 |
| --- | --- | --- | --- |
| 생성 대상 | 마지막 관측과 미래의 잔차, p.2 식 (1) | `future - last_input` | `target='residual'`; 직접 미래 생성은 별도 비교 |
| 손실 | epsilon 제곱오차, p.3 식 (4) | epsilon mean MSE | 동일; 별도 영상 손실·refiner를 섞지 않음 |
| diffusion step | DDPM [12] 설정, 추론 1000, p.4 III-A | 1000, linear beta 1e-4→0.02 | core와 같은 수식을 별도 설정 가능한 class로 구현·대조 |
| time condition | Fig.1의 정규화된 이산 grid | 정수 index를 sinusoidal embedding에 전달 | 기본 `time_input='index'`는 첨부 구현 보존; `'fraction'` 선택 가능 |
| 연속 gamma | 본문에 없음 | 사용하지 않음 | SCRD 기본은 `'timestep'`; DDPM 비교에서만 gamma 선택 |
| U-Net 입력 | history + noisy residual, `(L+P)C`, Fig.1 | 채널 concat | 같은 입력 구성, C와 L/P를 설정 가능 |
| STCG | spatial encoder + N개 TCU + output Conv, p.3 식 (5) | 4 scale, 각 TCU 3개 | 주요 배선 보존, 미기재 수치는 구현 선택으로 공개 |
| SWCI | Q/V=noise, K=cue, p.3 식 (6) | 식과 같은 source | 일반적인 K/V=condition으로 바꾸지 않음 |
| reverse variance | 구체 선택 미기재 | posterior `beta_tilde` | 기본 `'posterior'`, `'beta'`도 명시적으로 선택 가능 |
| 최종 복원 | residual + 마지막 관측, p.3 | 마지막 관측 1회 추가 | 같은 연산 후 물리 단위 복원 |
| 학습 | Adam, 초기 LR 2e-4, 250K iterations, p.4 | Adam, LR 1e-5, 80 epochs | 논문 LR/update 수를 기본으로 하되 변경 가능 |
| 데이터 | radar, 10분, 6→6, 128², p.3–4 | SSI, 30분, 5→4, 512² | 두 과제를 구분한 설정; 데이터가 같다고 주장하지 않음 |

Fig.1에는 4개의 SWCI Down/Up과 여러 scale의 STCG 주입이 보인다.
Fig.2(b)는 GN → time scale/shift → cross-attention을 두 번 거친 뒤 Conv/Deconv와
shortcut을 더한다. 세부 채널 수, stride, window 크기 등은 수치로 명시하지 않는다.
현재 구현을 논문 수식·그림에 맞게 재구성할 수 있지만 저자 코드와 완전히 같은 모델이라고
확인할 자료는 없다. 공식 코드와 supplement는 이번 검색에서 확인하지 못했다.
현재 네 scale 구현은 세 번 stride-2로 축소하며, deepest down block과 마지막 up block은
해상도를 유지한다. 이 세부 stride 배치는 첨부 코드에서 보존했고 논문에 수치로 명시되지는 않았다.

식 (5)의 ⊗를 저자는 Kronecker product라고 부르지만 세부 tensor shape는 적지 않았다.
현재 코드는 `G:[B,C,H,W]`, `D:[B,C,1,1]`의 channel broadcast 곱으로 gate를 만든다.
이는 첨부 구현과 TCU의 채널 가중치 해석을 유지한 것이며, 일반적인 두 행렬의
full Kronecker product를 계산했다는 뜻은 아니다.

## 3. 수식과 코드에서 의미가 같아야 하는 부분

`X`는 과거 L개 지도, `Y`는 미래 P개 지도다. 각 지도는 C개 변수를 가진다.
채널로 묶은 shape는 `X: [B,L*C,H,W]`, `Y: [B,P*C,H,W]`다.
`last`는 마지막 관측 C개 채널을 P개 미래에 반복한 값이다.

```python
clean = future - last             # SCRD residual target, Eq. (1)
noise = torch.randn_like(clean)
noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
pred_noise = model(noisy, time, history)
loss = (pred_noise - noise).square().mean()
```

위의 `alpha_bar`는 선택한 step의 누적곱을 `[B,1,1,1]`로 맞춘 값이다.
`alpha_t = 1-beta_t`, `alpha_bar_t = product(alpha_s, s<=t)`다.
코드 index `t=0,...,999`는 논문의 noise step `m=1,...,1000`에 대응한다.
`time_input='fraction'`이면 `(t+1)/num_diff_steps`의 이산 grid를 사용한다.
Fig.1은 이 grid를 그리지만 embedding 주파수와 구체 scaling은 적지 않았다.

역과정 평균은 다음 두 표현이 같다. clipping을 넣으면 이 동치가 깨질 수 있다.

```text
clean_hat = (noisy - sqrt(1-alpha_bar_t) * pred_noise) / sqrt(alpha_bar_t)
mean = posterior_mean(clean_hat, noisy, t)
     = (noisy - beta_t / sqrt(1-alpha_bar_t) * pred_noise) / sqrt(alpha_t)
```

분산은 `beta_t` 또는 `beta_tilde_t = beta_t*(1-alpha_bar_prev)/(1-alpha_bar_t)`다.
DDPM 원문 §3.2는 둘을 모두 다룬다. 기존 diffusionPack의 `p_sample`은 beta,
첨부 SCRD는 posterior 분산을 사용한다. 차이는 존재하지만 그 자체가 수식 오류는 아니다.
모듈 효과를 비교할 때는 같은 값을 선택해야 한다. 마지막 step에는 추가 noise가 없다.
최종 `future_hat = residual_hat + last` 후에만 동일한 scaler로 물리 단위를 복원한다.

입력과 미래가 각각 [-1,1]이어도 잔차는 [-2,2]일 수 있다.
따라서 잔차에 이미지용 [-1,1] clipping을 자동 적용하지 않는다.
현재 첨부 SCRD에는 이 clipping 오류가 없으며, 재구성 과정에서 새로 넣지 않도록 한 것이다.

## 4. 무엇을 baseline이라고 부르는가

| 비교 대상 | 설정·역할 |
| --- | --- |
| Persistence | 모든 미래를 마지막 관측으로 예측. 잔차의 기준이며 학습모델은 아님 |
| 기존 Prediction | main의 연속 gamma + AdaGN + 미래 공동 출력. 기존 프로젝트 결과와 연결하는 기준 |
| Direct DDPM | `backbone='ddpm'`, `target='future'`, `noise_conditioning='timestep'` |
| Residual DDPM | 같은 DDPM backbone에서 `target='residual'`; 잔차 표현의 영향 비교 |
| SCRD 전체 | `backbone='scrd'`, `target='residual'`, `noise_conditioning='timestep'`; STCG/SWCI 포함 |
| SCRD w/o RP | SCRD backbone을 유지하고 `target='future'`로 변경 |
| TCU 제거 비교 | spatial encoder와 output Conv를 유지하고 TCU gate를 제거; 이 재구성에서 정의한 ablation |

논문 Table I–II의 외부 비교모델은 ConvLSTM, MAU, SimVP, DiffCast, MCVD, PreDiff다.
MCVD가 비교표에 있다고 해서 SCRD backbone이 MCVD 전체 구현이라는 뜻은 아니다.
SCRD backbone은 저자가 제시한 SWCI U-Net이다.

논문 ablation은 `w/o STCG`, `w/o RP`다(p.5 III-C, Table I).
STCG를 없앴을 때 condition 경로를 어떻게 대체했는지는 본문에 없다.
재구성의 TCU 제거 비교를 저자의 exact `w/o STCG` 구현이라고 부르면 안 된다.
DDPM과 SCRD backbone의 parameter 수·capacity가 다르므로 둘의 성능 차이를
STCG 하나의 효과라고 결론내릴 수도 없다. 통제할 설정과 변경한 요소를 함께 기록한다.

## 5. 조절 가능한 설정과 근거

| 근거 구분 | 항목 |
| --- | --- |
| SCRD 본문 명시 | Adam, 초기 LR 2e-4, 250K optimizer updates, inference 1000 steps |
| SCRD 원 데이터 설정 | 과거6/미래6, 10분 간격, 128×128, 레이더 1변수 |
| DDPM [12]를 따른 값 | T=1000, linear beta 1e-4→0.02, epsilon objective |
| 현재 SSI 과제 설정 | 과거5/미래4, 30분 간격, +30/60/90/120분, 512×512, SSI 1변수 |
| 공개할 구현 선택 | widths, TCU depth, kernel/dilation, window/shift, GN groups, time embedding |
| 별도 결정할 학습 설정 | batch, Adam betas/eps, weight decay, scheduler, AMP, EMA, gradient clipping |
| 별도 결정할 데이터 설정 | scaler 범위·산출 근거, split, 결측 규칙, availability, resize 방식 |
| 별도 결정할 추론·평가 설정 | reverse variance, ensemble K, raw member와 ensemble mean의 평가 |

기존 모듈의 widths=(32,64,128,256), TCU depth=3, window=8, shift=4,
DW kernel3/dilation2, time dimension128, GN groups는 원문에 명시된 수치가 아니다.
그 값을 기본으로 유지하는 것은 현재 작업과의 연속성을 위한 재구성 선택이다.
현재 SSI 설정을 사용한 실험은 논문의 레이더 성능을 그대로 재현한 실험과 구분한다.
250K는 epoch 수가 아니므로 실제 optimizer update 수로 기록한다.

## 6. 기존 코드에서 확인한 문제와 아직 검증할 가설

**데이터 처리 차이:** `main/data.py`는 NaN/Inf를 0으로 바꾸고 범위 밖 값을 자른다.
새 경로는 CSV 기반 `ForecastDataset`을 공유하고 이런 조용한 대체·clipping을 넣지 않는다.
유효하지 않은 실제 관측에 대한 정책은 데이터 특성에 맞게 별도로 결정해야 한다.
새 모델의 개선이 관측되면 전처리 변화와 모듈 변화의 효과를 구분해야 한다.

**시간축·분할:** 기존 TXT는 첫 미래 시각을 뜻한다. 예를 들어 07:00이면
origin=06:30, 과거 04:30~06:30, 정답 07:00~08:30이다.
CSV는 각 관측의 `timestamp`, `path`, `available_at`을 명시한다.
실제 공개 지연을 모르는 상태에서 availability를 관측시각과 같게 만들면 새로운 가정이다.
미래 target을 다른 split과 공유하지 않고, cadence·결측·스케일링을 모델 간 동일하게 맞춘다.

**학습 재개:** 첨부 main은 optimizer/step을 저장하지만 일반 train 경로에서 다시 불러오지 않는다.
중단된 joint 학습을 그대로 이어가는 기능이 부족하다. 재구성에서는 모델·optimizer·step과
모델/확산/데이터 설정을 함께 저장하고 호환성을 확인하는 재개 경로가 필요하다.

**학습량·precision:** 기존 LR 1e-5는 논문 초기값보다 20배 작다.
80 epochs가 250K updates보다 많은지 적은지는 유효 표본 수와 batch로 계산해야 한다.
BF16 학습과 FP32 추론의 차이도 검사 대상이지만 실패 원인으로 입증되지는 않았다.
먼저 동일 precision으로 작은 학습·추론을 확인하고 변경을 별도 실험으로 기록한다.

**이미 수정된 내용:** 현 ZIP의 STCG gate 초기화와 output GN8은 이미 적용돼 있다.
이전 GN32 실험의 문제를 아직 남아 있는 버그로 다시 보고하지 않는다.
GN32/C32가 채널별 상수 shift를 제거하는 것은 확인했지만,
그 성질이 전체 모델의 밝기 실패를 일으켰다는 인과관계는 입증하지 못했다.

**평가 조건 차이:** 기존 validation은 제한된 30개 표본·K=1, test는 K=8 mean이다.
비교 시 동일 표본과 K를 사용하고, 개별 생성 member와 ensemble mean을 나누어 기록한다.
Raw MSE와 clipped SSIM/FID는 전처리가 다르다. 범위 밖 출력과 누락 member도 확인한다.
임의 epsilon bias 보정, fixed head, refiner를 논문 기본 모델에 자동 추가하지 않는다.

## 7. 검증된 범위와 후속 실험의 경계

이번에 실제로 실행한 검사는 다음과 같다.

| 검사 | 확인한 결과 | 의미·한계 |
| --- | --- | --- |
| DDPM forward/reverse 독립식 비교 | 최대 절대오차 약 7e-8 이하 | 작은 CPU tensor의 수식 일치 |
| 1000-step oracle sampling | 순서·마지막 noise 제거·조건 cache 확인 | 정답을 아는 oracle이며 학습 품질 아님 |
| 기존 SCRD 실제 backbone backward | 5→4, B=2, 16²에서 유한 gradient | 학습 수렴이나 SSI 성능 아님 |
| 기존 contract suite | 19개 중 16개 통과, 실제 자료 의존 3개 skip | 일부 평가 검사는 mock 사용 |
| 기존 diffusionPack + 새 SCRD 통합 tests | 전체 83개 통과; 최종 평가 유한값 검사 추가 뒤 runner 12개 재확인 | 단위·회귀·합성 CPU integration 검사 |
| 이식 backbone vs 기존 backbone | 16²와 17×19에서 동일 가중치의 출력 오차 0 | 이식 전후 수치 동치 |
| 저장·재개 vs 연속 학습 | 3 update와 2+1 update의 모델·Adam·RNG·loss 완전 일치 | 작은 CPU 합성 학습 |
| 새 실제 SCRD 학습·추론 | 네 scale, 5→4, 16², 4 updates 후 1000-step 생성 및 4개 지도 저장 | 수렴·예측 품질 검사 아님 |
| 새 noise-level 진단 | t=0/50/200/500/799/999에서 실행 | forward-noised 정답 복원이며 자유 생성 성능과 별개 |
| 기존 TXT → CSV의 표본 대응 | train 8/valid 9/test 9개의 origin·과거·미래가 동일 | 유한하고 범위 안에 있는 합성 fixture; 실제 SSI 전체는 미실행 |
| 실제 SSI 재학습·추론·성능 비교 | 수행하지 않음 | 데이터·실패 checkpoint·GPU가 이번 자료에 없음 |

4-step smoke test는 호출 경로만 확인한다. beta endpoint를 그대로 둔 4-step에서는
`alpha_bar_T≈0.9603`으로 신호가 거의 남으므로 standard Gaussian 출발의 품질 검증이 아니다.
1000-step에서는 `alpha_bar_T≈4.04e-5`다. 이때 epsilon의 작은 평균오차도 clean 복원에서
증폭될 수 있으므로 낮은 epsilon MSE만으로 미래 지도 성능을 판단하지 않는다.

실제 비교에서는 persistence, direct DDPM, residual DDPM, SCRD를 같은 split/scaler/
schedule/variance/학습 budget으로 실행하고, lead별 평균편향·MAE/MSE와 생성 member를 살핀다.
그 결과와 실패 checkpoint의 동일 조건 재실행이 있어야 과거 성능 문제의 원인을 좁힐 수 있다.

## 출처

- [SCRD 원문 DOI](https://doi.org/10.1109/LGRS.2024.3486112): 첨부 PDF p.2 식 (1)–(2), Fig.1; p.3 식 (3)–(6), Fig.2; p.4 III-A와 Tables I–II; p.5 III-C.
- [DDPM 원문](https://arxiv.org/html/2006.11239v2): §3.2, Algorithms 1–2, §4. 두 variance 선택 및 linear schedule의 근거.
- [TAU 원문](https://openaccess.thecvf.com/content/CVPR2023/html/Tan_Temporal_Attention_Unit_Towards_Efficient_Spatiotemporal_Predictive_Learning_CVPR_2023_paper.html): SCRD 참고문헌 [16]. Gate 구조의 참고이며 TAU 전체 predictor/loss를 SCRD로 가져오라는 지시는 아님.
