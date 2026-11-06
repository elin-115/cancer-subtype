# 진단서 (Gene + 단서)

유전자 변이 정보로 암 아형(SUBCLASS, 26종)을 분류하는 프로젝트입니다.
OZ 코딩스쿨 × 데이콘 해커톤 1조 (2026.11.06 ~ 11.13)

## 팀
김지연 · 김도엽 · 여은지 · 김새봄 · 박종부

## 문제
- 입력: 환자 한 명의 유전자별 변이 정보 (`WT` 또는 `V600E`, `R132H`, `S269fs` 같은 변이 표기)
- 출력: 암 아형 (예: `BRCA`, `KIPAN`, `SKCM`) + 상위 3개 아형 확률
- 평가: Macro F1

## 폴더 구조
```
cancer-subtype/
├── (노트북 추가 예정)
├── requirements.txt
├── data/               # 대회 데이터 (git에 올리지 않음)
└── output/             # 제출 파일, 모델 (git에 올리지 않음)
```

## 실행 방법
1. 데이콘에서 받은 `train.csv`, `test.csv`, `sample_submission.csv`를 `data/` 폴더에 넣습니다.
2. 라이브러리를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
   macOS에서 xgboost가 `libomp.dylib` 오류를 내면 `brew install libomp`를 실행합니다.
3. 노트북을 위에서부터 실행합니다. 데이터 경로는 첫 셀의 `DATA_DIR`에서 바꿉니다. (노트북은 정리 후 추가 예정)

## 결과
| 모델 | 피처 | CV Macro F1 | 리더보드 |
|---|---|---|---|
| 베이스라인 | 이진 변이 4384개 + 변이 개수 | 0.4069 | - |

## 개발 환경
macOS 15.7 (arm64), Python 3.14.6

## 규칙
- 전처리 기준(인코더, 통계, 피처 선택)은 train에서만 만들고, test에는 적용만 합니다.
- 대회 데이터와 API 키, 비밀번호는 올리지 않습니다.
