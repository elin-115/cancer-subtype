# ④ 치료 타깃 변이 train 집계표

작성: 김도엽 (2026-11-09). [4_0_맞춤치료_종합.md](./4_0_맞춤치료_종합.md) 4절과 [4_2](./4_2_맞춤치료_표적치료근거_김도엽.md) 2절의 변이를 한 표로 합친 것입니다.

- train 6,201명만 사용(test 미사용). 셀을 공백으로 나눈 뒤 토큰이 **정확히 일치**하는 행의 수(한 행에 중복 표기돼도 1).
- 단위: `정확` = 정확한 변이 토큰, `위치` = 해당 번호 자리의 모든 변이, `유형` = `*`(nonsense) 또는 `fs`(frameshift) 포함 토큰.
- 표기 빈도일 뿐 독립 환자 수·병원성 판정 결과가 아닙니다. 미국 허가와 국내 허가·급여는 다르며 "미확인"은 이번 조사에서 확인하지 못한 항목입니다.
- 저장소 규칙(*.csv 미업로드)에 따라 표로 올립니다. 원본 CSV는 팀 드라이브에 둡니다.

| 유전자 | 변이 | 단위 | train 환자 수 | SUBCLASS 분포 | 치료 예시 | 미국 FDA | 국내(식약처·급여) | 허가 판단에 더 필요한 정보(데이터에 없음) |
|---|---|---|---|---|---|---|---|---|
| BRAF | V600E | 정확 | 321 | THCA 184; SKCM 106; COAD 20; LUAD 5; GBMLGG 4; LGG 1; KIPAN 1 | dabrafenib+trametinib(암종불문, CRC 제외) / encorafenib+cetuximab(+화학요법)·CRC | US FDA: 암종불문 2022(가속), CRC 병용 2026-02-24(정식) | KR: 흑색종·NSCLC 허가, 암종불문 미확인; 비라토비(CRC 2차) 2024-01 급여 | 암종·병기·이전치료·승인검사 |
| BRAF | V600K | 정확 | 15 | SKCM 15 | 흑색종 BRAF+MEK 억제제 | US FDA: 흑색종 허가(V600E/K) | KR: 흑색종 허가(V600E/K) | 병기·절제 가능성 |
| BRAF | 600번 전체 | 위치 | 339 | THCA 184; SKCM 124; COAD 20; LUAD 5; GBMLGG 4; LGG 1; KIPAN 1 | (참고) V600E 외 변이는 위 허가 대상이 아님 | — | — | 정확 변이 확인 |
| IDH1 | R132H | 정확 | 345 | GBMLGG 171; LGG 166; LAML 5; PRAD 2; BLCA 1 | vorasidenib(2등급 성상세포종·희돌기교종, 수술 후) / ivosidenib(AML) | US FDA: vorasidenib 2024-08, ivosidenib AML | KR: 보라니고 2026-01-13 허가, 2026-05 비급여 출시, 급여 심사 중(미확인) | WHO 조직형·등급·수술 여부·승인검사 |
| IDH1 | R132C | 정확 | 46 | SKCM 13; LAML 11; GBMLGG 9; LGG 8; LIHC 2; BRCA 1; THYM 1; COAD 1 | 위와 동일(감수성 IDH1 변이) | US FDA: 동일 | KR: 동일 | 암종(LAML vs 신경교종)별 조건 다름 |
| IDH1 | 132번 전체 | 위치 | 411 | GBMLGG 189; LGG 180; LAML 18; SKCM 14; LIHC 3; PRAD 2; BRCA 1; THYM 1 | — | — | — | 정확 변이 확인 |
| IDH2 | R140Q | 정확 | 13 | LAML 13 | enasidenib(재발/불응 AML) | US FDA: 2017-08 | KR: 미확인 | 재발/불응 여부·승인검사 |
| IDH2 | R172K | 정확 | 14 | GBMLGG 6; LGG 5; LAML 3 | enasidenib(AML) / vorasidenib(신경교종) | US FDA: 각각 허가 | KR: 미확인 / 보라니고 | 암종별 조건 다름 |
| PIK3CA | H1047R | 정확 | 147 | BRCA 109; STES 12; UCEC 8; COAD 5; HNSC 4; LUSC 2; GBMLGG 2; CESC 1 | alpelisib+fulvestrant / inavolisib+palbociclib+fulvestrant (HR+/HER2− 유방암) | US FDA: alpelisib 2019, inavolisib 2024-10 | KR: 이토베비 2025-07-29 허가(급여 미확인); 피크레이 허가(급여 미확인) | HR/HER2 상태·내분비치료 이력·승인검사 |
| PIK3CA | E545K | 정확 | 149 | BRCA 48; HNSC 16; STES 15; CESC 15; LUSC 10; BLCA 9; UCEC 9; COAD 8 | 동일 | 동일 | 동일 | 동일 |
| PIK3CA | E542K | 정확 | 75 | BRCA 30; HNSC 13; CESC 8; UCEC 5; BLCA 4; LUSC 3; STES 3; LGG 2 | 동일 | 동일 | 동일 | 동일 |
| EGFR | L858R | 정확 | 7 | LUAD 7 | osimertinib 등 EGFR-TKI (NSCLC) | US FDA: 허가 | KR: 타그리소 허가·급여(조건) | 조직형·병기·치료 단계 |
| EGFR | T790M | 정확 | 1 | LUAD 1 | osimertinib (EGFR-TKI 후 내성) | US FDA: 허가 | KR: 허가 | 이전 TKI 노출 |
| FGFR3 | S249C | 정확 | 11 | BLCA 8; LUSC 2; HNSC 1 | erdafitinib (요로상피암) | US FDA: 2019(가속)→2024(정식) | KR: 미확인 | 조직형·병기·이전치료·동반진단 |
| ERBB2 | S310F | 정확 | 11 | STES 6; CESC 2; OV 1; BLCA 1; BRCA 1 | (점변이) 표준 허가 없음 — HER2 증폭/IHC3+와 다름 | US FDA: T-DXd는 IHC3+ 기준(2024, 암종불문) | KR: 미확인 | HER2 IHC/증폭 검사 |
| RET | M918T | 정확 | 2 | PCPG 2 | selpercatinib(RET 변이 MTC) — PCPG 라벨과 맥락 다름 | US FDA: MTC 허가; RET fusion 암종불문 2026-07(정식) | KR: 레테브모 급여 확대 2026-10(세부 미확인) | 조직형(MTC 여부)·fusion 여부 |
| KIT | D816V | 정확 | 5 | LAML 3; TGCT 2 | (참고) 비만세포증 avapritinib; GIST 1차 약제 내성 변이 | US FDA: 비만세포증 허가 | KR: 미확인 | 질환 맥락 |
| HRAS | Q61R | 정확 | 24 | PCPG 11; THCA 9; PRAD 3; LUSC 1 | (참고) tipifarnib 연구 단계(HNSC) | 연구 단계 | — | — |
| AKT1 | E17K | 정확 | 8 | PRAD 2; THCA 2; HNSC 1; SKCM 1; STES 1; LUSC 1 | capivasertib(HR+/HER2− 유방암, AKT1/PIK3CA/PTEN) | US FDA: 2023-11 | KR: 미확인 | HR/HER2·치료 이력 |
| NPM1 | WQ288fs | 정확 | 21 | LAML 21 | (참고) AML 위험도 분류 마커; 직접 표적치료제보다 진단·예후 | — | — | — |
| BRCA1 | 절단(*/fs) | 유형 | 22 | OV 8; STES 4; BRCA 3; SKCM 2; LUSC 2; UCEC 2; COAD 1 | PARP 억제제 (난소·유방·전립선·췌장, 조건부) | US FDA: 암종·치료단계별 허가 | KR: 암종별 허가·급여 상이 | 병원성 판정·생식세포/체세포·암종·치료 단계 |
| BRCA2 | 절단(*/fs) | 유형 | 38 | STES 9; OV 6; UCEC 4; BLCA 4; COAD 3; SKCM 2; LUSC 2; BRCA 2 | 동일 | 동일 | 동일 | 동일 |
| PTEN | 절단(*/fs) | 유형 | 209 | UCEC 85; GBMLGG 30; BRCA 14; STES 13; KIPAN 13; SKCM 9; LUSC 8; KIRC 8 | (참고) capivasertib 대상 변이 중 하나(유방암) | US FDA: 2023-11 | KR: 미확인 | HR/HER2·치료 이력 |
| MLH1 | 절단(*/fs) | 유형 | 14 | STES 4; SKCM 2; UCEC 1; LUSC 1; ACC 1; BLCA 1; THYM 1; OV 1 | (참고) dMMR 단서일 뿐 MSI/dMMR 검사 결과 아님 | — | — | MSI/IHC 검사 |
| MSH2 | 절단(*/fs) | 유형 | 18 | UCEC 4; STES 4; SKCM 2; PRAD 1; ACC 1; OV 1; CESC 1; THYM 1 | 동일 | — | — | 동일 |
| PMS2 | 절단(*/fs) | 유형 | 6 | COAD 2; SKCM 1; STES 1; LUSC 1; UCEC 1 | 동일 | — | — | 동일 |
| POLE | P286R | 정확 | 10 | UCEC 8; PAAD 1; BRCA 1 | (참고) 초변이 표현형 단서; 면역치료 연구 | 연구 단계 | — | TMB/MSI 검사 |
