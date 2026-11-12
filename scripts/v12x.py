"""도엽 v12(10a_v12_model_only.ipynb)를 그대로 옮기고 피처 실험 옵션을 붙인 스크립트.

옵션 (인자로 여러 개):
  SIG     원인별 흔적 묶음 (치환쌍을 CpG·자외선·담배·APOBEC 4묶음으로 합친 비율·개수)
  MSI     복구 고장(MSI) 흔적: 반복 서열 표적 유전자 8개의 frameshift 0/1·개수, 침묵 변이 뺀 frameshift 비율
  TWO     두 번 맞은 유전자 (한 칸에 서로 다른 변이 2개 이상) 0/1, fold 안 10명 이상
  PAIRW   유전자 짝 단어: 자주 바뀌는 유전자 100개끼리 'A+B' 단어를 TF-IDF에 추가 (텍스트 bigram)
  LSA     TF-IDF를 SVD로 100개 주제로 압축해 XGB에 추가 (토픽 모델)
  LGBM    XGB 대신 LightGBM / LGBM3: XGB·LR·LGBM 1:1:1
  KNN     TF-IDF 코사인 최근접 환자 15명 투표를 0.2 비중으로 섞기
  SUBMIT  전체 train 학습 후 output/submission_<이름>.csv 저장 (형제 라벨 없음)

사용: python scripts/v12x.py <이름> [SIG] [TWO] [SUBMIT]
"""
import os, re, sys, time
from collections import Counter, defaultdict
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
import warnings; warnings.filterwarnings('ignore')

NAME, OPTS = sys.argv[1], set(sys.argv[2:])
DATA_DIR, OUT_DIR, SEED, HOT_N, W_XGB, LR_C, STAGE2_C = 'data', 'output', 42, 5, 0.5, 10.0, 0.005

train = pd.read_csv(f'{DATA_DIR}/train.csv', dtype=str); test = pd.read_csv(f'{DATA_DIR}/test.csv', dtype=str)
submission = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
genes = [c for c in train.columns if c not in ('ID', 'SUBCLASS')]; G = np.array(genes)
V_raw = train[genes].fillna('WT').values
Vt_raw = test.reindex(columns=genes, fill_value='WT').fillna('WT').values
le = LabelEncoder(); y = le.fit_transform(train['SUBCLASS']); CL = list(le.classes_); N = len(y)

# ---- 표기 통일 (v12 그대로)
def sig(t): return re.sub(r'\d+', '', t)
HOT_CNT = Counter()
for row in V_raw:
    for j in np.flatnonzero(row != 'WT'):
        for h in {f'{G[j]}_{n[0]}' for n in (re.findall(r'\d+', m) for m in row[j].split()) if n}: HOT_CNT[h] += 1

def normalize(V):
    out = V.copy(); r, c = np.nonzero(V != 'WT')
    for i, j in zip(r, c):
        groups = defaultdict(list)
        for t in V[i, j].split():
            if re.search(r'del|ins|dup', t): continue
            t = re.sub(r'^([A-Z]\d+)X$', r'\1*', t); t = re.sub(r'^([A-Z]\d+)[A-Z]fs$', r'\1fs', t)
            groups[sig(t)].append(t)
        keep = []
        for ts in groups.values():
            if len(set(ts)) > 1:
                keep.append(max(ts, key=lambda t: HOT_CNT.get(f'{G[j]}_{(re.findall(r"[0-9]+", t) or ["x"])[0]}', 0)))
            else: keep += ts
        out[i, j] = ' '.join(keep) if keep else 'WT'
    return out
V, Vt = normalize(V_raw), normalize(Vt_raw)

# ---- 피처
AA = list('ACDEFGHIKLMNPQRSTVWY') + ['*']; AI = {a: i for i, a in enumerate(AA)}
MS = re.compile(r'^([A-Z])(\d+)([A-Z*])$')
# 원인별 흔적: 그 원인의 대표 염기 변화로 생기는 아미노산 변화 (Alexandrov 시그니처 기반 근사)
SIGS = {'cpg': ['RC', 'RH', 'RW', 'RQ', 'R*'], 'uv': ['SF', 'PL', 'PS'], 'tobacco': ['GV', 'GC', 'DY', 'E*'], 'apobec': ['EK', 'EQ']}
# MSI 표적 유전자: 반복 서열이 있어 복구 고장 시 frameshift가 잘 생기는 유전자 (암종 구분 없는 목록)
MSI_G = ['ACVR2A', 'BAX', 'CASP5', 'JAK1', 'PRDM2', 'TCF7L2', 'TGFBR2', 'PTEN']; MSI_J = {g: k for k, g in enumerate(MSI_G)}
SIG_IX = {k: [AI[p[0]] * 21 + AI[p[1]] for p in v] for k, v in SIGS.items()}

def featurize(V):
    n = len(V); Cm = np.zeros((n, 443), np.float32); hot, words, two = [], [], []; MF = np.zeros((n, len(MSI_G)), np.float32)
    for i, row in enumerate(V):
        h, w, tw = [], [], []
        for j in np.flatnonzero(row != 'WT'):
            g = G[j]; w.append(g); toks = list(dict.fromkeys(row[j].split()))
            if len(toks) >= 2: tw.append(g)
            for t in toks:
                m = MS.match(t)
                if m and m.group(1) in AI and m.group(3) in AI: Cm[i, AI[m.group(1)] * 21 + AI[m.group(3)]] += 1
                elif 'fs' in t:
                    Cm[i, 441] += 1
                    if g in MSI_J: MF[i, MSI_J[g]] = 1
                else: Cm[i, 442] += 1
                w.append(f'{g}_{t}'); nums = re.findall(r'\d+', t)
                if nums: h.append(f'{g}_{nums[0]}'); w.append(f'{g}@{nums[0]}')
        hot.append(list(dict.fromkeys(h))); words.append(w); two.append(tw)
    BIN = sp.csr_matrix((V != 'WT').astype(np.float32)); tot = Cm.sum(1, keepdims=True)
    SUB = np.hstack([Cm / np.maximum(tot, 1), np.log1p(Cm)]).astype(np.float32)
    S4 = np.stack([Cm[:, ix].sum(1) for ix in SIG_IX.values()], 1)
    SIGF = np.hstack([S4 / np.maximum(tot, 1), np.log1p(S4)]).astype(np.float32)
    silent = Cm[:, [k * 21 + k for k in range(20)]].sum(1, keepdims=True); nons = np.maximum(tot - silent, 1)
    MSIF = np.hstack([MF, MF.sum(1, keepdims=True), Cm[:, 441:442] / nons, np.log1p(Cm[:, 441:442])]).astype(np.float32)
    return dict(MSI=MSIF, BIN=BIN, NMUT=np.asarray(BIN.sum(1)).ravel(), HOT=hot, WORDS=words, SUB=SUB, SIG=SIGF, TWO=two)
TR, TE = featurize(V), featurize(Vt)
if 'PAIRW' in OPTS:   # 라벨 안 씀, train 변이 빈도로만 고름
    TOPG = set(G[np.argsort(-np.asarray(TR['BIN'].sum(0)).ravel())[:100]])
    for S_ in (TR, TE):
        for w in S_['WORDS']:
            gs = sorted({t for t in w if t in TOPG}); w += [f'{a}+{b}' for k, a in enumerate(gs) for b in gs[k + 1:]]
KEY = pd.util.hash_pandas_object(pd.DataFrame(V_raw), index=False).values.astype(np.int64)
GRP = KEY.copy(); z = np.flatnonzero(TR['NMUT'] == 0); GRP[z] = -(z + 1)
def identity(d): return d
def sub(L, ix): return [L[i] for i in ix]

# ---- 환자 중복 정리 (v8A)
def find_pairs(ix, min_pairs=20):
    labs = defaultdict(set)
    for i in ix: labs[GRP[i]].add(y[i])
    cnt = Counter(tuple(sorted(s)) for s in labs.values() if len(s) == 2); tot = Counter(y[ix]); pairs = {}
    for (a, b), c in cnt.items():
        if c >= min_pairs: spec, umb = (a, b) if tot[a] < tot[b] else (b, a); pairs[umb] = spec
    return pairs
def clean(ix, pairs):
    g = defaultdict(list)
    for i in ix: g[GRP[i]].append(i)
    keep, lab = [], []
    for rows in g.values():
        ls = {y[i] for i in rows}; spec = [pairs[u] for u in ls if u in pairs and pairs[u] in ls] if len(ls) > 1 else []
        if spec: keep.append(rows[0]); lab.append(spec[0])
        else: keep += rows; lab += [y[i] for i in rows]
    return np.array(keep), np.array(lab)

XGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.3, tree_method='hist', n_jobs=-1, random_state=SEED)

class Model:
    def fit(self, ix, lab, pairs):
        self.pairs = pairs
        self.hot = CountVectorizer(analyzer=identity, min_df=HOT_N, binary=True, lowercase=False).fit(sub(TR['HOT'], ix))
        if 'TWO' in OPTS: self.two = CountVectorizer(analyzer=identity, min_df=10, binary=True, lowercase=False).fit(sub(TR['TWO'], ix))
        self.cv = CountVectorizer(analyzer=identity, min_df=2, binary=True, lowercase=False).fit(sub(TR['WORDS'], ix))
        self.tf = TfidfTransformer(sublinear_tf=True).fit(self.cv.transform(sub(TR['WORDS'], ix)))
        Xl = self._Xl(TR, ix); sw = compute_sample_weight('balanced', lab)
        if 'LSA' in OPTS: self.svd = TruncatedSVD(100, random_state=SEED).fit(Xl)
        Xx = self._Xx(TR, ix); self.ms = []
        if 'LGBM' not in OPTS: self.ms.append(XGBClassifier(**XGB_PARAMS).fit(Xx, lab, sample_weight=sw))
        if 'LGBM' in OPTS or 'LGBM3' in OPTS: self.ms.append(LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3, subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1).fit(Xx, lab, sample_weight=sw))
        self.lr = LogisticRegression(C=LR_C, class_weight='balanced', max_iter=3000).fit(Xl, lab)
        if 'KNN' in OPTS: self.knn = KNeighborsClassifier(15, metric='cosine', algorithm='brute', weights='distance').fit(Xl, lab)
        self.gmask = np.flatnonzero(np.asarray(TR['BIN'][ix].sum(0)).ravel() >= 5); self.s2 = {}
        for umb, spec in pairs.items():
            m = np.isin(lab, [umb, spec])
            self.s2[(umb, spec)] = LogisticRegression(C=STAGE2_C, class_weight='balanced', max_iter=2000).fit(self._X2(TR, ix[m]), (lab[m] == spec).astype(int))
        return self
    def _Xx(self, S, ix):
        parts = [S['BIN'][ix], sp.csr_matrix(S['NMUT'][ix][:, None].astype(np.float32)), self.hot.transform(sub(S['HOT'], ix)), sp.csr_matrix(S['SUB'][ix])]
        if 'SIG' in OPTS: parts.append(sp.csr_matrix(S['SIG'][ix]))
        if 'MSI' in OPTS: parts.append(sp.csr_matrix(S['MSI'][ix]))
        if 'TWO' in OPTS: parts.append(self.two.transform(sub(S['TWO'], ix)))
        if 'LSA' in OPTS: parts.append(sp.csr_matrix(self.svd.transform(self._Xl(S, ix)).astype(np.float32)))
        return sp.hstack(parts, format='csr').astype(np.float32)
    def _Xl(self, S, ix): return self.tf.transform(self.cv.transform(sub(S['WORDS'], ix)))
    def _X2(self, S, ix): return sp.hstack([S['BIN'][ix][:, self.gmask], sp.csr_matrix(np.log1p(S['NMUT'][ix])[:, None])], format='csr')
    def predict(self, S, ix):
        Xx, Xl = self._Xx(S, ix), self._Xl(S, ix)
        Ptree = [m.predict_proba(Xx) for m in self.ms]; Plr = self.lr.predict_proba(Xl)
        P = (sum(Ptree) + Plr) / 3 if len(Ptree) == 2 else W_XGB * Ptree[0] + (1 - W_XGB) * Plr
        if 'KNN' in OPTS:
            Pk = np.zeros_like(P); Pk[:, self.knn.classes_] = self.knn.predict_proba(Xl); P = 0.8 * P + 0.2 * Pk
        Pm = P.copy()
        for umb, spec in self.s2: Pm[:, umb] = P[:, umb] + P[:, spec]; Pm[:, spec] = -1
        pred = Pm.argmax(1)
        for (umb, spec), mdl in self.s2.items():
            m = pred == umb
            if m.any(): pred[m] = np.where(mdl.predict_proba(self._X2(S, ix[m]))[:, 1] >= 0.5, spec, umb)
        return pred

def fit_model(ix):
    pairs = find_pairs(ix); k, l = clean(ix, pairs); return Model().fit(k, l, pairs)

ALL = np.arange(N); pred = np.zeros(N, int); t0 = time.time(); fs = []
for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(ALL, y), 1):
    pred[va] = fit_model(tr).predict(TR, va); fs.append(f1_score(y[va], pred[va], average='macro'))
print(f'{NAME} {sorted(OPTS)} | folds', ' '.join(f'{v:.4f}' for v in fs), f'| OOF {f1_score(y, pred, average="macro"):.4f} | {time.time() - t0:.0f}s', flush=True)
np.save(f'{OUT_DIR}/oof_{NAME}.npy', pred)

if 'SUBMIT' in OPTS:
    p = fit_model(ALL).predict(TE, np.arange(len(Vt)))
    s = submission.copy(); s['SUBCLASS'] = le.inverse_transform(p); s.to_csv(f'{OUT_DIR}/submission_{NAME}.csv', index=False, encoding='utf-8-sig')
    print('saved', f'{OUT_DIR}/submission_{NAME}.csv')
