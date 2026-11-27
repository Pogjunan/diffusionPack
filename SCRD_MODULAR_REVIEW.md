# SCRD 모듈 분리 및 VSCode 실행 전 점검

2026-11-26. 기준은 직전에 제공한 `diffusionPack_prediction_scrd.zip`이다. 첨부 `SCRD(1).zip`의 원본 `main/module.py`, SCRD 논문 p.2 Fig.1·식 (1)–(2), p.3 Fig.2·식 (3)–(6), p.4 실험 설정과 비교했다. pstack-research-codex의 research policy 및 research-mode, diffusion-audit, math-code-audit, redteam, training-guard를 적용했다. pstack 설치 스크립트나 외부 자동화는 실행하지 않았다.

## 코드 변경

- `diffusionpack/scrd/stcg.py`: SpatialEncoder, TCU, STCG.
- `diffusionpack/scrd/swci.py`: SWCrossAttention, SWCIBlock.
- `diffusionpack/scrd/unet.py`: SinusoidalTimeEmbedding, SWCIUNet, SCRDUNet.
- `diffusionpack/scrd/prediction.py`: 기존 residual loss와 sampling 연결. 공통 `diffusion.ddpm_loss`, `main.sampling.p_sample_loop`를 사용한다.
- `diffusionpack/scrd/comparison.py`: 기존 독립 비교 엔진(분산·time-input·target 변경 실험용). 일반 Prediction 경로에서 이 엔진으로 sampling하지 않는다. 구형 `scrd.py`는 패키지로 대체했다.
- 기존 `backbone/scrd_unet.py`, `prediction.py`의 SCRD import 이름은 호환용 import로 유지한다. 구현은 중복하지 않는다. state_dict의 key와 tensor shape는 변경하지 않았다.
- `main/check_prediction.py`: 실제 ForecastDataset과 loss를 이용하는 preflight. 사용 가능한 train/validation/test metadata 확인, 제공된 split 간 target 중복 및 시간 순서 검사, 작은 학습 batch의 loss와 gradient 검사. test grid의 값은 읽지 않는다. 실제 데이터는 수정하지 않는다. 엄격한 시간 분할 검사를 통과하지 않으면 기존 split을 임의로 변경하지 않고 이유를 검토해야 한다.
- `main/train_prediction.py`: 새 학습이 기존 출력 폴더를 덮어쓰지 못하게 차단한다. 기존 출력 폴더의 `latest.pt`에서 명시적으로 재개하는 것은 가능하다. 공통 Dataset 구성 함수는 preflight와 공유한다.
- `run_prediction.py`: 기본 `mode="check"`. 학습이나 추론은 모드를 명시적으로 변경한 뒤 실행한다.

## 연구 계약

현재 SSI 예측 예시는 history=5, future=4, C=1, 512×512, cadence=30분, 첫 lead=30분, 마지막 관측과 발행 사이 lag=0으로 기존 설정을 보존했다. history 및 future는 각각 L*C, P*C 순서의 채널 축에 결합된다. 실제 데이터의 물리 단위, scaling bounds, availability 및 split은 제공되지 않았으므로 미확인이다. 사용자가 실제 자료에 맞춰 입력해야 한다. 논문은 128×128 radar, 6→6, 10분 간격, Adam 초기 LR 2e-4, 250K iterations, 추론 1000 steps다. 현재 SSI 실험과 동일하지 않다.

SCRD에서 R0=Y-X_last, q(R_t|R0)=N(sqrt(alpha_bar_t)R0,(1-alpha_bar_t)I), loss=mean((eps_pred-eps)^2), 최종 Yhat=Rhat+X_last. 현재 Prediction의 공통 DDPM path는 정수 index 0..999, linear beta 1e-4..0.02, beta variance, residual clipping 없음. 별도 비교 엔진에는 posterior variance 및 fraction time 옵션도 남긴다. 논문의 식 (6)에 따라 Q/V는 noise feature, K는 STCG cue. 식 (5)는 기존 구현의 channel-broadcast 곱으로 해석했고 일반적인 full Kronecker product와 동일하다고 주장하지 않는다. 정수 time embedding, 채널 폭, window/TCU 개수, GN 설정은 기존 구현을 보존했다. 논문에 모든 세부 값이 명시되지는 않는다. `config.py`, `diffusion.py`, DDPM/Restore U-Net 및 공통 train loop는 변경하지 않았다.

## 이번에 실제로 실행한 검사

환경: PyTorch 2.10.0+cpu, pandas 2.2.3, CUDA 없음. 모두 합성 데이터 또는 정적 코드 검사다.

| 검사 | 상태 | 근거 및 한계 |
|---|---|---|
| 분리 전 기존 suite | PASS | 91 tests |
| 분리 후 기존 suite | PASS | 91 tests |
| 추가 검사 포함 전체 suite | PASS | 99 tests; `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q` |
| 모델 보존 | PASS | 5→4, 16², 기본 4 scale; 동일 seed의 모든 state_dict 값 및 같은 입력 출력, 모든 파라미터 gradient가 bit-exact; 첨부 SCRD 원본 ForecastDenoiser에도 strict load하고 동일 출력 확인 |
| 식 (5) TCU | PASS | 2채널, 수동 gate 값의 독립 결과와 정확히 일치. channel gate를 생략한 negative control은 불일치 |
| 식 (6) SWCI | PASS | 좌표 기반 독립 reference, float64 atol/rtol=1e-12; shift/padding/작은 spatial 축 4개 사례, B=2. V를 condition으로 바꾸거나 mask를 제거한 negative control 불일치 |
| STCG/SWCI 연결 | PASS | 네 scale shape, cache 사용 여부 동일 출력, cue 교체에 따른 출력 변화, 모든 gradient 존재 및 유한성 |
| DDPM | PASS | 기존 forward/mean/variance 독립 수치·oracle 검사; 1000-step 공통 sampler 호출 및 final step, residual 복원 검사 포함 |
| 학습 및 추론 연결 | PASS | `python -m tools.smoke_prediction_scrd --output <new-dir>`; 실제 4 scale, 16² CPU, 3 optimizer updates(중간 저장 후 재개 포함), 1000-step sampling, 4개 finite NPY 저장. 15.64초. 재개는 기능 검증이지 exact RNG 복원이 아니다. |
| 128² 실제 계산 | PASS | 기본 widths, B=1, FP32 CPU forward/backward, loss=1.133207 및 모든 gradient 유한 |
| 512² 실제 계산 | PASS | 기본 widths, B=1, FP32 CPU forward만. 출력 (1,4,512,512), finite. 512² backward 및 GPU 메모리는 미측정 |
| 데이터 누출 검사 | PASS (합성) | 새 preflight의 target 중복·시간 순서 negative control, 기존 history availability 검사; pstack audit-splits의 실제 ForecastDataset에서 추출한 48개 lead 행 통과. 잘못된 availability 한 행은 exit 2로 거부됨. 실제 운영 availability 정확성이나 실제 split 무누출을 입증하지 않음 |
| checkpoint 보호 | PASS | 기존 latest.pt에 새 학습 시도 시 거부, sentinel bytes 불변. 이전 checkpoint 키와 모델 구조 유지 |
| pstack 도구 자체 | PASS | `python .agents/research/tests/test_research_guard.py -q`: 24 tests OK. 앞선 pytest 호출은 timeout되어 직접 unittest로 다시 실행했다. |
| 실제 SSI 성능 및 실패 원인 | NOT RUN / INCONCLUSIVE | NPY 데이터, 원래 split, 실패 checkpoint, GPU 없음. 작은 unconditional 생성 성공만으로 prediction 품질을 판단하지 않는다. |

## VSCode 실행 순서

1. ZIP을 별도 폴더에 풀고 그 폴더를 연다. 기존 체크포인트/결과를 덮어쓰지 않는다. 필요시 `pip install -e .` (설치된 torch/CUDA 버전 확인).
2. `python -m pytest -q` 및 `python -m tools.smoke_prediction_scrd --output runs/scrd_smoke_new`로 로컬 환경에서 재검사한다.
3. `run_prediction.py`의 train_config에서 실제 `csv_path`, `target_csv`(공용 frame CSV를 사용하는 경우), `value_min`, `value_max`, `units`, cadence/lag/lead를 설정한다. `valid_csv`, `valid_targets`, `test_csv`, `test_targets`도 실제 split에 맞춰 설정한다. 기본값 `None`이면 해당 split 점검은 생략되므로 완전한 누출 검사가 아니다. `check_all=True`로 train/valid의 모든 사용 grid 검사 가능. 실제 관측 availability가 맞는지는 CSV 출처에서 별도 확인한다.
4. `mode="check"`로 `python run_prediction.py`. 기본 B=1; 필요하면 `check_batch_size`를 실제 batch에 맞춰 GPU 메모리/gradient를 다시 검사한다. 이 모드는 optimizer update 및 파일 저장이 없다. AMP 선택도 train 설정과 동일하게 확인한다.
5. 실제 실험은 `mode="train"`으로 설정한 뒤 새로운 output 경로에서 짧은 steps로 시작한다. 정상 저장 확인 후 동일 설정으로 resume. `mode="inference"`에서 checkpoint와 weights를 확인하고, 검증용 split을 사용해 실제 지도 생성 및 성능을 확인한다. 테스트는 모델 선택에 사용하지 않는다. 기존 Prediction trainer는 자동 best-validation checkpoint 선택을 수행하지 않는다. 별도 `run_scrd.py`는 validation 및 비교 진단용 경로다.

## 남은 검증

실제 SSI의 train-only scaling, split/availability, 512² GPU forward/backward 및 장기 training, validation에서 noise level별 epsilon MSE 및 clean reconstruction bias, persistence 대비 성능, ensemble 및 lead별 평가가 필요하다. `backbone="ddpm"`은 gamma+AdaGN direct, `backbone="scrd"`는 discrete residual+STCG/SWCI이므로 두 모델의 차이를 단일 module 효과로 해석할 수 없다. `run_scrd.py`의 비교 모드에서 공통 timestep 및 variance, 같은 데이터·budget·seed로 direct/residual/SCRD 차이를 분리한다. 이전 실패 checkpoint가 제공되면 당시 설정을 복원하고 원인을 진단한다. GitHub push는 수행하지 않았다. 논문 완독이나 논문 성능 재현을 의미하지 않는다.
