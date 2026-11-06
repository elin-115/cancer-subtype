# Data Leakage 정리

데이콘 안내(Data Leakage 관련 안내, Data Leakage 실수 사례집)를 정리하고, 우리 프로젝트에 어떻게 적용하는지 적은 문서입니다.
Data Leakage가 적발되면 **실격**이고 수상 대상에서도 제외되므로, 피처나 모델을 추가할 때마다 이 문서를 기준으로 확인합니다.

## 1. 기본 전제

- test 데이터는 **전혀 알 수 없고 볼 수 없다**는 전제로 진행합니다. 실제 서비스에서는 들어올 환자 데이터를 미리 알 수 없기 때문입니다.
- train은 학습과 검증용, test는 **예측만 하는 대상**입니다. 인코딩 기준, 통계, 컬럼 구조를 정할 때 test를 보면 안 됩니다.
- 외부 데이터는 사용할 수 없습니다. 논문으로 공개되고 법적 제약이 없는 사전 학습 모델은 사용할 수 있습니다.

## 2. Leakage에 해당하는 경우

| 구분 | 예시 |
|---|---|
| 인코딩 | test로 LabelEncoder를 fit, train과 test를 합쳐서 인코딩 |
| get_dummies | test에 `pd.get_dummies()` 적용, `align()`으로 train과 test 컬럼 맞추기 |
| 스케일링 | test로 Scaler를 fit |
| 결측치 | test의 평균, 최빈값 등으로 결측치 채우기 |
| 파생변수 | 2개 이상의 test 행 정보를 사용, test의 이동 평균, test 분포를 보고 구간 나누기 |
| 학습 | test를 모델 학습에 사용 |
| 행 정보 | test의 index나 행 개수 정보 활용 |
| EDA | test를 EDA해서 얻은 인사이트를 전처리나 학습에 반영 |
| 군집·피처 선택 | train과 test를 같이 보고 clustering이나 feature selection 수행 |

## 3. Leakage에 해당하지 않는 경우

- **한 행(환자 한 명) 안에서만 하는 연산**과 파생변수. 예: `test['C'] = test['A'] + test['B']`
- **train의 통계**로 test를 전처리하는 것. 예: train의 평균으로 test 결측치 채우기
- train에서 만든 인코더나 기준을 test에 **transform만** 하는 것

## 4. 검증(CV)에서 조심할 것

규칙 위반은 아니지만, 지키지 않으면 CV 점수가 실제보다 좋게 나와서 잘못된 판단을 하게 됩니다.

- Scaler, Imputer, Encoder도 **모델의 일부**로 봅니다. fold가 나뉘면 fold 안의 train 부분으로 다시 fit합니다.
- **피처 선택**(예: 중요한 유전자 상위 N개)도 fold 안에서 train 부분으로만 고릅니다.
- **빈도·타깃 기반 피처**(예: "train에서 10번 이상 나온 변이만 사용", 아형별 비율)도 fold 안에서 계산합니다.
- 최종 학습 때는 train 전체로 기준을 만들고, test에는 그 기준을 적용만 합니다.

```python
# 잘못된 예: fold를 나누기 전에 전체 데이터로 fit
X = scaler.fit_transform(X)
for tr, va in skf.split(X, y):
    model.fit(X[tr], y[tr])

# 올바른 예: fold마다 train 부분으로만 fit
for tr, va in skf.split(X, y):
    scaler.fit(X[tr])
    X_tr = scaler.transform(X[tr])
    X_va = scaler.transform(X[va])
    model.fit(X_tr, y[tr])
```

## 5. 우리 베이스라인 점검 결과

| 항목 | 처리 방식 | 판정 |
|---|---|---|
| 라벨 인코딩 | `LabelEncoder`를 train의 `SUBCLASS`에만 fit | ✅ |
| 유전자 컬럼 목록 | `gene_cols`는 train에서만 만들고, test는 `reindex`로 맞춤. train에 없는 값은 `WT`로 채움 | ✅ |
| 변이 피처 | `WT`면 0, 그 외는 1. 행마다 따로 계산하고 학습하는 값 없음 | ✅ |
| 변이 개수 `n_mut` | 한 환자 행 안에서 합계 | ✅ |
| 클래스 가중치 | `compute_sample_weight('balanced')`를 fold마다 `y[tr_idx]`로 계산 | ✅ |
| `concat` / `get_dummies` / `align` | 사용하지 않음 | ✅ |
| test 사용 | 불러오기 → `make_features` → `predict`만 수행 | ✅ |
| 추론 함수 | `predict_one()`은 환자 한 명만 넣어도 같은 결과가 나옴 | ✅ |

최종 노트북에서는 `print(test.shape)`처럼 test를 들여다보는 출력도 빼서, test를 예측에만 썼다는 것을 분명히 합니다.

## 6. 팀 규칙

1. 피처는 `make_features(df, gene_cols)` 안에서 **행 단위로만** 만듭니다.
2. 데이터에서 무언가를 학습해야 하는 피처는 `fit` / `transform` 함수를 따로 만들고, CV 루프 안에서 호출합니다.
3. test는 EDA하지 않습니다. 분포 확인, 값 개수 세기, 시각화 모두 train에서만 합니다.
4. 공개 코드는 아이디어만 참고하고 복사하지 않습니다. 기존 대회 코드와 90% 이상 유사하면 수상 대상에서 제외됩니다.
5. 판단이 애매하면 코드를 합치기 전에 팀에 공유하고, 필요하면 운영진에게 문의합니다.

## 7. 제출 전 체크리스트

- [ ] 코드에 `concat(train, test)` 후 `fit`하는 부분이 없다
- [ ] test에 `get_dummies`, `align`, `fit`, `fit_transform`을 쓰지 않았다
- [ ] Scaler, Imputer, Encoder, 피처 선택은 CV fold 안에서 fit했다
- [ ] test의 통계, 행 개수, index를 사용하지 않았다
- [ ] 외부 데이터 파일을 불러오지 않았다
- [ ] 새 커널에서 처음부터 끝까지 오류 없이 실행된다

## 8. 운영진 확인 필요

- 논문에 공개된 유전자·신호경로 목록(예: 특정 경로에 속한 유전자 이름)을 참고해서 파생변수를 만드는 것이 외부 데이터 사용에 해당하는지 확인합니다. 별도 파일을 불러오지 않고, 유전자 이름만 코드에 직접 적는 방식입니다.
  - 답변: (대기 중)
