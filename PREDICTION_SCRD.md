# 기존 Prediction 경로에 SCRD 연결

이 파일은 앞선 별도 SCRD runner 다음에 추가한 연결을 설명한다.
지금은 `run_prediction.py` → `train_prediction()` / `infer_prediction()`에서도 SCRD를 선택할 수 있다.
기존 DDPM의 loss와 sampler를 실제 호출하므로 이 경로에서는 새 diffusion engine을 만들지 않는다.

## 1. 연결 구조

| 부분 | 사용 코드 |
| --- | --- |
| 데이터 | 공통 `ForecastDataset`, history `[B,L*C,H,W]`, future `[B,P*C,H,W]` |
| 생성 타깃 | 마지막 관측 C채널을 P번 반복해 `future - last` |
| Noise predictor | `scrd/unet.py`의 SCRDUNet, `scrd/stcg.py`, `scrd/swci.py` |
| 학습 | `scrd.prediction.scrd_prediction_loss` → 기존 `diffusion.ddpm_loss` → 기존 `q_sample` |
| 추론 | `scrd.prediction.scrd_prediction_sample` → 기존 `p_sample_loop` → 기존 `p_sample` |
| 미래 복원 | 생성 residual에 마지막 관측을 한 번 추가 |

학습과 추론 모두 기존 1000-step schedule과 정수 t를 사용한다.
condition feature는 추론 시작 시 한 번 계산한다. residual을 [-1,1]로 clipping하지 않는다.
`diffusionpack/diffusion.py`, `config.py`, DDPM U-Net, 공통 `main/train.py`는 이번 연결에서도 그대로다.

**이 통합 경로의 reverse variance는 기존 core와 같은 beta다.**
첨부 SCRD의 posterior variance와 다르며, SCRD 원문은 이 선택을 구체적으로 지정하지 않는다.
앞서 만든 `run_scrd.py`는 beta/posterior를 선택할 수 있는 별도 비교·진단 경로로 남아 있다.
두 경로가 동일한 학습·추론 설정이라고 가정하지 않는다.

## 2. 실행 설정

`run_prediction.py`의 예시를 SCRD 5→4, 512², Adam LR 2e-4, 250000 updates로 설정했다.
먼저 이 파일에서 실제 CSV 경로, value_min/value_max, units를 입력한다.
공용 frames.csv를 쓰면 해당 split의 `target_csv`도 반드시 지정한다.
`mode="check"`로 데이터 및 실제 학습 배치의 forward/backward를 먼저 검사한다. `valid_csv`, `test_csv`와 각 target 목록을 지정하면 split 경계도 점검한다. `check_all=True`는 사용되는 모든 train/valid grid를 한 번씩 읽는다. 검사 후 `mode="train"`으로 학습하고, `mode="inference"`로 저장한 모델에서 추론한다.

```bash
python run_prediction.py
```

직접 Python에서 사용할 때도 같은 설정 객체를 쓴다.

```python
config = PredictionTrainConfig(
    backbone="scrd",
    history_steps=5,
    future_steps=4,
    resolution=512,
    steps=250000,
    epochs=None,
    warmup_steps=0,
    amp=False,
)
# Set config.csv_path, value_min, value_max, and units for the actual data.
checkpoint = train_prediction(config)
```

`PredictionTrainConfig()` 자체의 기본 backbone은 기존 `ddpm`이다.
`backbone="ddpm"`의 gamma + AdaGN 동작 및 이전 gamma checkpoint는 유지한다.
SCRD와 DDPM checkpoint는 서로 바꾸어 불러오지 않는다.

SCRD의 widths/window/TCU/time dimension/GN은 `scrd_*` 설정으로 조절한다.
이 값들은 model_config에 저장하며, 추론에서는 현재 기본값 대신 checkpoint 설정을 사용한다.
목표 split의 모든 미래 시각이 target_csv에 포함된 window만 사용한다.
추론에는 미래 정답 배열이 없어도 된다.

## 3. 공통 trainer를 사용할 때의 차이

기존 trainer는 Adam, scheduler, EMA를 저장한다. 위 예시는 warmup=0, AMP off,
추론 weights="model"로 초기 비교를 하도록 설정했다. EMA도 별도로 계산·저장되지만
원문이 EMA를 명시했다는 뜻은 아니다. 필요하면 추론에서 weights="ema"를 따로 비교한다.

기존 trainer의 resume은 optimizer/scheduler/model/EMA를 복원한다.
RNG와 epoch 내 sample 위치까지 복원하는 방식은 아니므로, 이 경로의 재개를
중단 없는 학습과 bit-exact하다고 주장하지 않는다.
앞선 `run_scrd.py`에서 확인한 exact resume 결과는 그 별도 runner에만 해당한다.

## 4. 검증 순서

코드 연결은 한 작업 안에서 구현할 수 있다. 검사는 단계별로 진행한다.

1. 기존 DDPM 함수를 실제 사용하는지, residual/정수 t/shape가 맞는지 확인.
2. STCG/SWCI의 gradient와 condition cache, checkpoint 복원을 확인.
3. 작은 합성 데이터에서 학습·재개·1000-step 추론과 미래 지도 저장을 실행.
4. 실제 자료에서는 같은 split·전처리·budget으로 기존 Prediction과 SCRD를 비교.

1–3은 실행 경로 검증이다. 4의 성능·과거 실패 원인은 실제 자료와 실패 checkpoint가 있어야 판단한다.

이번 연결 후 전체 테스트는 91개가 통과했다. 새 통합 검사 8개에는 기존 core 함수의
실제 호출, 1000-step oracle, 과거 gamma checkpoint 호환, SCRD 저장·재개·weight 선택이 포함된다.
별도로 실제 네 scale SCRD를 기존 Prediction trainer에서 3 updates 학습하고,
16² 합성 데이터의 과거 5장으로부터 1000-step 추론해 미래 4장을 저장했다.
이 실행은 품질 검증이나 실제 SSI 학습 결과가 아니다.

```bash
python -m tools.smoke_prediction_scrd --output runs/prediction_scrd_smoke
```

배포 ZIP은 앞선 재구성 전체를 포함한다. 함께 제공한 `diffusionPack_prediction_scrd.patch`는
직전에 받은 `diffusionPack_scrd_rebuild.zip`에 추가로 적용하는 패치이며, GitHub main용 누적 패치가 아니다.

## 5. 최신 모듈 분리와 검증

2026-11-26 후속 작업은 [SCRD_MODULAR_REVIEW.md](SCRD_MODULAR_REVIEW.md)를 참고한다.
이 문서의 91개 테스트 기록은 분리 전 통합본에 대한 기록이다.
