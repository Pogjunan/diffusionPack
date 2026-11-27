# SCRD 학습·추론 실행

기존 `run_prediction.py`에 SCRD를 직접 선택하는 최신 연결은 [PREDICTION_SCRD.md](PREDICTION_SCRD.md)에 있다.
이 문서는 앞서 만든 별도 `run_scrd.py` 비교·진단 경로의 사용법이다.

논문과 기존 코드의 대조 결과는 [SCRD_REVIEW.md](SCRD_REVIEW.md)에 있다.
현재 첨부 코드의 DDPM 식 자체가 잘못됐다고 확인되지는 않았다.
이번 작업은 같은 데이터 처리·DDPM 수식으로 원인을 나눠 확인할 수 있는 재구성이다.

기준 소스는 GitHub main `0e819b0bc685c76c8ba510f1e4f769629b29e028`과
이전에 준비한 CSV 정리 패치다. 기존 `diffusionpack/config.py`, `diffusion.py`,
DDPM U-Net 및 기존 Prediction/Restore의 모델 코드는 수정하지 않았다.
새 학습 경로의 체크포인트는 기존 SCRD/Prediction 체크포인트와 별도 형식이다.

## 1. 먼저 수정할 설정

`diffusionpack/scrd_config.py`를 직접 수정한다.

| 설정 | 의미 |
| --- | --- |
| `train_csv`, `valid_csv`, `test_csv` | `timestamp,path,available_at` 열이 있는 CSV |
| `*_targets` | 선택 사항. 해당 split에서 허용한 정답 시각만 담은 CSV |
| `value_min`, `value_max`, `units` | 실제 저장된 값의 스케일과 단위. 세 값 모두 확인해서 지정 |
| `history_steps`, `future_steps`, `map_channels` | 기본 5장 → 4장, 지도당 1변수 |
| `resolution`, `cadence_minutes`, `lead_minutes` | 기본 512, 30분 간격, 첫 미래 +30분 |
| `learning_rate`, `steps`, `batch_size` | 기본 2e-4, 250000 optimizer updates, batch16 |
| `widths`, `window`, `tcu_depth`, `time_dim`, GN 설정 | 논문에 수치가 없는 구현 선택. 기존 SCRD 값을 기본으로 보존 |
| `diffusion_steps`, `beta_start`, `beta_end` | 기본 1000, 1e-4, 0.02의 linear schedule |
| `reverse_variance` | 기본 posterior; 기존 diffusionPack과 맞추려면 beta |
| `time_input` | index: 기존 코드 보존. fraction: Fig.1의 이산 `(t+1)/T` 표현 |
| `precision` | 기본 float32. bfloat16은 지원 CUDA GPU에서 별도 선택 |
| `validation_every`, `validation_origins`, `validation_samples` | 기본 10000 updates마다 30 origins, K=1 |

Scalers는 실제 학습 자료나 정해진 물리 기준으로 설정한다.
기존 SCRD config에는 0~26.347150802612가 있지만, 그 값의 산출 근거·단위는
첨부 ZIP만으로 확인하지 못했다. 이 값이나 W/m2를 새 기본값으로 확정하지 않았다.
NaN/Inf는 오류로 알리고, 범위 밖 유한값은 자동으로 자르지 않는다.

논문의 레이더 과제 차원을 사용할 때는 `history_steps=6`, `future_steps=6`,
`resolution=128`, `cadence_minutes=10`, `lead_minutes=10`으로 바꾼다.
이 설정만으로 레이더 데이터·전처리·저자 실험 전체가 재현되는 것은 아니다.
입력 배열의 실제 크기가 설정과 같아야 하며 이 코드가 자동 resize하지 않는다.

## 2. 기존 TXT 분할을 그대로 연결하기

이미 ForecastDataset용 CSV가 있으면 이 단계를 건너뛴다.
기존 SCRD의 TXT는 **첫 정답 지도 시각**을 담고 있다.
`prepare_scrd_data.py`는 이를 별도의 target CSV로 옮긴다.
모든 미래 정답 시각이 해당 split 목록 안에 있는 origin만 사용하므로
기존 공동 출력의 target 분할 규칙을 유지한다. 과거 관측은 공유할 수 있다.

아래 시간대와 공개 지연은 예시다. 파일명이 KST이고 관측 즉시 사용 가능하다고
확인된 자료에서만 `+09:00`, `0`을 사용한다. 파일명이 UTC이면 `+00:00`이다.

```bash
python prepare_scrd_data.py \
  --data-dir /your/data/npy \
  --train-txt /your/data/train.txt \
  --valid-txt /your/data/valid.txt \
  --test-txt /your/data/test.txt \
  --time-offset +09:00 --available-delay-minutes 0 \
  --output data/scrd_csv
```

기본 파일명은 `2026-11-26_1230.npy` 형식이다.
다른 형식이면 `--filename-format`을 지정한다.
설정에는 다음 경로를 연결한다.

```python
train_csv = valid_csv = test_csv = "data/scrd_csv/frames.csv"
train_targets = "data/scrd_csv/train_targets.csv"
valid_targets = "data/scrd_csv/valid_targets.csv"
test_targets = "data/scrd_csv/test_targets.csv"
```

`frames.csv`의 timestamp는 관측 시각, available_at은 사용할 수 있게 된 시각이다.
공개 지연이 있으면 `history_lag_minutes`와 첫 목표까지의 `lead_minutes`도
실제 예측 원점에 맞춰 정한다. UTC 변환이 공개 지연을 대신하지는 않는다.
중복 시각, split 간 정답 중복, 목록의 정답 파일 누락은 변환 단계에서 알린다.

## 3. 설치와 데이터 확인

자신의 GPU에 맞는 PyTorch를 설치한 환경에서 실행한다.

```bash
python -m pip install -e .
python run_scrd.py check-data
```

`check-data`는 모든 사용 가능한 train/valid window의 shape와 유한값을 읽어 확인한다.
데이터 양이 크면 시간이 걸리지만 학습이나 test 점수 계산은 하지 않는다.
제외된 window 수와 train/valid 사이의 history 겹침을 출력한다.
공유된 history는 기존 방식과 같이 허용하며, target 중복은 거부한다.
CSV 없이 임의로 split을 새로 나누지 않는다.

## 4. 같은 조건에서 비교하는 모델

공통 데이터, budget, precision, variance를 유지하고 아래 설정을 각각 다른 폴더에 실행한다.
이 명령들은 설정된 전체 학습을 시작한다. 기본 budget은 250000 updates다.

```bash
# 1) Integer-time DDPM: future itself.
python run_scrd.py train --backbone ddpm --target future --output runs/direct_ddpm

# 2) Same DDPM U-Net: future minus the last observation.
python run_scrd.py train --backbone ddpm --target residual --output runs/residual_ddpm

# 3) Residual + STCG/SWCI U-Net.
python run_scrd.py train --backbone scrd --target residual --output runs/scrd
```

SCRD w/o residual은 `--backbone scrd --target future`로 실행한다.
`use_tcu=False`는 TCU gate만 제거한다. 논문의 exact w/o STCG라고 부르지 않는다.
DDPM과 SCRD는 parameter 수와 backbone도 다르므로, 둘의 차이를 STCG 하나의 효과로 해석하지 않는다.

기존 gamma+AdaGN Prediction을 새 비교 경로에서 실행하려면
`--backbone ddpm --target future --noise-conditioning gamma --reverse-variance beta`를 사용한다.
모델과 loss 연결은 기존 구현을 재사용하지만, 새 runner는 EMA/warmup 없이 Adam으로 학습한다.
기존 Prediction의 전체 학습 절차·체크포인트와 동일한 실행이라고 보지는 않는다.
기존 `run_prediction.py` 경로도 그대로 남아 있다.
공용 `frames.csv`를 쓰면 각 경로에 split target 목록을 지정한다.
`run_scrd.py`에서는 `*_targets`, `run_prediction.py`에서는 `target_csv` 설정을 사용한다.

## 5. 저장, 재개, 추론

`latest.pt`에는 모델·Adam·완료 update·epoch 내 위치·RNG·설정·사용 origin을 저장한다.
`best.pt`는 고정 validation origins와 동일 seed/K의 ensemble-mean MSE가 개선될 때 저장한다.
MSE는 물리 단위에서 계산하고 각 origin/horizon에 같은 가중치를 준다.
`validation_every=0` 또는 valid_csv가 없으면 best.pt는 생성되지 않는다.
모든 validation에서 1000-step 생성을 하므로 비용은 origins×K에 비례한다.

```bash
# Total target is 250000 updates, not another 250000.
python run_scrd.py train --resume runs/scrd/latest.pt --output runs/scrd --steps 250000

# No future truth is needed for inference.
python run_scrd.py infer --checkpoint runs/scrd/best.pt \
  --test-csv data/scrd_csv/frames.csv --test-targets data/scrd_csv/test_targets.csv \
  --samples 8 --max-origins 1 --output runs/scrd/forecasts

# Evaluate all selected test origins only after fixing the experimental choices.
python run_scrd.py evaluate --checkpoint runs/scrd/best.pt \
  --all-origins --samples 8 --output runs/scrd/test_results

# No checkpoint is needed. Use the same scaler/splits in scrd_config.py.
python run_scrd.py persistence --all-origins --output runs/persistence
```

Resume은 모델·noise·scale·단위·학습 설정과 사용 origin이 같아야 한다.
CSV/NPY 내용 자체를 해시로 검사하지 않으므로 같은 파일의 값을 중간에 바꾸지 않는다.
기존 출력 폴더에서는 같은 run의 latest.pt만 이어 쓸 수 있다.
다른 checkpoint로 분기할 때는 새 빈 폴더를 사용한다. 이때 이전 best.pt가 자동 복사되지는 않는다.
분기한 run에서 기존 best보다 좋아진 checkpoint가 없으면 latest.pt나 원래 run의 best.pt를 명시한다.

추론의 architecture, schedule, scale, cadence는 checkpoint에서 읽는다.
새 설정 파일의 다른 값으로 checkpoint 의미를 조용히 바꾸지 않는다.
`forecasts.csv`에는 origin/목표시각/member/파일/단위를 기록하고,
각 `.npy`는 `[C,H,W]`다. 같은 member 번호의 네 파일이 한 공동 생성 결과다.
평가에는 ensemble-mean MSE/MAE/bias/centered MSE와 개별 member MSE 평균이 나온다.
raw output을 clipping해서 지표를 좋게 만들지 않는다.
이 지표들은 디버깅용이며 논문의 CSI/HSS/CRPS/LPIPS/FID 전체 평가를 이식한 것은 아니다.

## 6. noise level별로 문제 좁히기

```bash
python run_scrd.py diagnose --checkpoint runs/scrd/latest.pt \
  --max-origins 4 --output runs/scrd/noise_check
```

validation 자료에서 t=0/50/200/500/799/999의 epsilon MSE·평균편향과
clean residual 복원 MSE·평균편향을 저장한다. 이 진단은 forward-noised 정답을 사용하는
복원 검사이고, 순수 Gaussian에서 시작하는 자유 생성 성능과 구분한다.
schedule을 바꾸면 같은 비율의 step을 사용한다.

## 7. 이번 실행을 다시 확인하기

```bash
python -m pip install pytest
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m tools.smoke_scrd --output runs/scrd_smoke
```

Smoke는 실제 네 scale SCRD, 5→4, 16×16 합성 데이터, 4 optimizer updates,
저장·재개·noise 진단·1000-step 추론을 실행한다. 결과는 합성 실행 검증이며 품질 검증이 아니다.
512×512 SSI의 장기 GPU 학습, 실패 checkpoint 비교, 논문 지표 재현은 별도로 수행해야 한다.
