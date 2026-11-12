"""아형별 판정 기준 조정 실험 (v12x.py가 저장한 OOF 확률 사용).

아형마다 확률에 곱할 가중치를 OOF 확률 위에서 맞춰 Macro F1을 올리는지 봄.
과적합을 막기 위해 OOF 행을 다시 5등분해, 4/5에서 가중치를 맞추고 남은 1/5에서 평가함 (중첩 교차검증).

사용: python scripts/tune_bias.py <v12x 실험 이름>
"""
import sys, glob, re
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

NAME = sys.argv[1]
y = LabelEncoder().fit_transform(pd.read_csv('data/train.csv', usecols=['SUBCLASS']).SUBCLASS)
P = np.load(f'output/oofP_{NAME}.npy'); base = np.load(f'output/oof_{NAME}.npy'); K = P.shape[1]
Q = {}
for f in glob.glob(f'output/oofQ_{NAME}_*_*.npy'):
    m = re.fullmatch(rf'output/oofQ_{re.escape(NAME)}_(\d+)_(\d+)\.npy', f)   # 다른 실험(이름_s7 등) 파일 섞이지 않게
    if m: Q[(int(m.group(1)), int(m.group(2)))] = np.load(f)

def decide(P, w, rows):
    Pw = P[rows] * w; Pm = Pw.copy()
    for u, s_ in Q: Pm[:, u] = Pw[:, u] + Pw[:, s_]; Pm[:, s_] = -1
    pred = Pm.argmax(1)
    for (u, s_), q in Q.items():
        m = pred == u; pred[m] = np.where(q[rows][m] >= 0.5, s_, u)
    return pred

ALLR = np.arange(len(y)); ones = np.ones(K)
assert (decide(P, ones, ALLR) == base).all(), 'OOF 판정 재현 실패'

def fit_w(rows, grid=(0.6, 0.75, 0.9, 1.1, 1.3, 1.6), sweeps=2):
    w = ones.copy(); best = f1_score(y[rows], decide(P, w, rows), average='macro')
    for _ in range(sweeps):
        for k in range(K):
            for g in grid:
                w2 = w.copy(); w2[k] *= g; sc = f1_score(y[rows], decide(P, w2, rows), average='macro')
                if sc > best + 1e-4: best, w = sc, w2
    return w

out = np.zeros(len(y), int)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=123).split(ALLR, y):
    w = fit_w(tr); out[te] = decide(P, w, te)
print(f'{NAME} | 조정 전 OOF {f1_score(y, base, average="macro"):.4f} → 중첩 조정 후 {f1_score(y, out, average="macro"):.4f}')

# 전체 OOF로 다시 맞춘 가중치 저장 (제출용). 사용: python scripts/tune_bias.py <이름> SAVE
if len(sys.argv) > 2 and sys.argv[2] == 'SAVE':
    w = fit_w(ALLR); np.save(f'output/bias_{NAME}.npy', w)
    print('saved', f'output/bias_{NAME}.npy', '| 전체 OOF 적용 시', f'{f1_score(y, decide(P, w, ALLR), average="macro"):.4f}', '| 바뀐 아형', {k: round(float(v), 2) for k, v in enumerate(w) if v != 1})
