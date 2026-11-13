"""데모용 추론: 최종 모델(v12 + STACK, 시드 3개 평균, 리더보드 0.4072)로 환자 한 명을 예측함.

- 모델 파일: `python scripts/v12x.py v12_stackS3 STACK SUBMIT EXPORT` (SKIPCV=1 SEEDS=42,7,123)로 만든 output/model_v12_stackS3.pkl
- 계산 방식은 scripts/v12x.py와 같음: 표기 통일 → 피처 → 시드마다 XGB·TF-IDF LR 확률 → 메타 LR → 시드 평균 → 2단계 판정
- 형제 라벨(train과 같은 변이 대조)은 쓰지 않음 → 실제 환자에게도 그대로 쓸 수 있음
"""
import re
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import joblib
import xgboost as xgb

AA = list('ACDEFGHIKLMNPQRSTVWY') + ['*']; AI = {a: i for i, a in enumerate(AA)}   # 아미노산 21개(끊김 * 포함)
MS = re.compile(r'^([A-Z])(\d+)([A-Z*])$')                                          # 한 글자 치환 예: V600E, R175*


class OncovaV12:
    def __init__(self, path='output/model_v12_stackS3.pkl'):
        art = joblib.load(path)
        self.genes = list(art['gene_cols']); self.G = np.array(self.genes); self.gi = {g: j for j, g in enumerate(self.genes)}
        self.classes = list(art['classes']); self.hot_cnt = art['hot_cnt']; self.models = art['models']
        self.sub_names = [f'sub_{a}>{b}' for a in AA for b in AA] + ['sub_fs', 'sub_etc']
        self.sub_names = [f'ratio_{n}' for n in self.sub_names] + [f'log_{n}' for n in self.sub_names]

    # ---- 표기 통일 (v12x.normalize와 같음): X→*, Nfs→fs, del/ins 제외, 같은 변이를 여러 위치로 적은 것은 train에서 흔한 위치 하나만
    def normalize_cell(self, g, cell):
        groups = defaultdict(list)
        for t in cell.split():
            if re.search(r'del|ins|dup', t): continue
            t = re.sub(r'^([A-Z]\d+)X$', r'\1*', t); t = re.sub(r'^([A-Z]\d+)[A-Z]fs$', r'\1fs', t)
            groups[re.sub(r'\d+', '', t)].append(t)
        keep = []
        for ts in groups.values():
            if len(set(ts)) > 1: keep.append(max(ts, key=lambda t: self.hot_cnt.get(f'{g}_{(re.findall(r"[0-9]+", t) or ["x"])[0]}', 0)))
            else: keep += ts
        return ' '.join(keep) if keep else 'WT'

    # ---- 피처 (v12x.featurize를 한 명용으로): 변이 유전자 0/1, 변이 수, 핫스팟 단어, 치환쌍 비율·개수, TF-IDF 단어
    def featurize(self, muts):
        known = {g: self.normalize_cell(g, c) for g, c in muts.items() if g in self.gi}
        known = {g: c for g, c in known.items() if c != 'WT'}
        b = np.zeros(len(self.genes), np.float32); Cm = np.zeros(443, np.float32); hot, words = [], []
        for g in sorted(known, key=self.gi.get):                        # 학습 때처럼 유전자 열 순서대로
            b[self.gi[g]] = 1; words.append(g)
            for t in dict.fromkeys(known[g].split()):
                m = MS.match(t)
                if m and m.group(1) in AI and m.group(3) in AI: Cm[AI[m.group(1)] * 21 + AI[m.group(3)]] += 1
                elif 'fs' in t: Cm[441] += 1
                else: Cm[442] += 1
                words.append(f'{g}_{t}'); nums = re.findall(r'\d+', t)
                if nums: hot.append(f'{g}_{nums[0]}'); words.append(f'{g}@{nums[0]}')
        sub = np.concatenate([Cm / max(Cm.sum(), 1), np.log1p(Cm)])
        return dict(known=known, BIN=b, NMUT=b.sum(), HOT=set(hot), WORDS=words, SUB=sub)

    def _xx(self, f, M):   # XGB 입력: 0/1 + 변이 수 + 핫스팟(이 모델 사전) + 치환쌍
        h = np.zeros(len(M['hot_vocab']), np.float32)
        for w in f['HOT']:
            if w in M['hot_vocab']: h[M['hot_vocab'][w]] = 1
        return sp.csr_matrix(np.concatenate([f['BIN'], [f['NMUT']], h, f['SUB']])[None].astype(np.float32))

    def _xl(self, f, M):   # LR 입력: 단어 있음 1 × idf, 길이 1로 정규화 (TfidfTransformer sublinear + l2와 같음)
        x = np.zeros(len(M['word_vocab']))
        for w in f['WORDS']:
            if w in M['word_vocab']: x[M['word_vocab'][w]] = M['idf'][M['word_vocab'][w]]
        n = np.linalg.norm(x); return sp.csr_matrix((x / n if n else x)[None])

    def _full(self, m, X):   # 모델이 아는 아형만 확률을 내므로 26칸에 맞춰 넣음
        P = np.zeros(len(self.classes)); P[m.classes_] = m.predict_proba(X)[0]; return P

    def predict(self, muts):
        """muts: {'BRAF': 'V600E', ...} → (화면용 확률 26개, 최종 1등 번호, 피처 dict)"""
        f = self.featurize(muts); Ps, Qs = [], defaultdict(list)
        for M in self.models:
            Xx, Xl = self._xx(f, M), self._xl(f, M)
            Z = np.hstack([self._full(M['xgb'], Xx), self._full(M['lr'], Xl)])[None]   # 두 모델 확률 52개
            Ps.append(self._full(M['meta'], Z))                                        # 메타 LR이 아형별 비중으로 섞음
            x2 = sp.csr_matrix(np.concatenate([f['BIN'][M['gmask']], [np.log1p(f['NMUT'])]])[None])
            for k, lr in M['s2'].items(): Qs[k].append(lr.predict_proba(x2)[0, 1])    # 묶음 안 세부 아형일 확률
        P = np.mean(Ps, 0); Q = {k: float(np.mean(v)) for k, v in Qs.items()}          # 시드 평균
        merged, proba = P.copy(), P.copy()
        for (umb, spec), q in Q.items():
            grp = P[umb] + P[spec]; merged[umb], merged[spec] = grp, -1; proba[spec], proba[umb] = grp * q, grp * (1 - q)
        best = int(merged.argmax())
        for (umb, spec), q in Q.items():
            if best == umb: best = spec if q >= 0.5 else umb                          # 제출과 같은 2단계 판정
        return proba, best, f

    def evidence(self, f, cls, k=3):
        """시드 42 XGB의 기여도로, 환자가 가진 변이 중 1등 아형 쪽으로 민 것 상위 k개 (원인 유전자라는 뜻은 아님)"""
        M = self.models[0]; X = self._xx(f, M); booster = M['xgb'].get_booster()
        if cls not in list(M['xgb'].classes_): return []
        c = booster.predict(xgb.DMatrix(X), pred_contribs=True)[0][list(M['xgb'].classes_).index(cls), :-1]
        hot_names = sorted(M['hot_vocab'], key=M['hot_vocab'].get)
        names = self.genes + ['n_mut'] + hot_names + self.sub_names
        x = X.toarray()[0]
        order = [i for i in np.argsort(-c) if x[i] > 0 and c[i] > 0 and names[i] != 'n_mut' and not names[i].startswith(('ratio_', 'log_'))]
        return [names[i] for i in order[:k]]
