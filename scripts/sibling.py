"""형제 라벨 후처리 (팀 결정, docs/00 A-6 판단 보류 상태).

같은 환자가 train에는 KIRC로, test에는 KIPAN으로 나뉘어 들어간 경우를 이용함.
test 행과 변이가 완전히 같은 train 행이 딱 하나이고 그 라벨이 형제 쌍 중 하나면,
test 행은 같은 환자의 다른 기록으로 보고 형제 라벨로 바꿈.

사용: python scripts/sibling.py output/submission_v8A_PAIR.csv [최소변이수=2]
결과: 입력 이름 뒤에 _SIB 붙인 파일
"""
import sys, collections
import pandas as pd

SIB = {'KIPAN': 'KIRC', 'KIRC': 'KIPAN', 'GBMLGG': 'LGG', 'LGG': 'GBMLGG'}
src = sys.argv[1]
min_mut = int(sys.argv[2]) if len(sys.argv) > 2 else 2  # 변이가 너무 적으면 다른 환자끼리도 우연히 같음

tr = pd.read_csv('data/train.csv').fillna('WT')
te = pd.read_csv('data/test.csv').fillna('WT')
genes = [c for c in tr.columns if c not in ('ID', 'SUBCLASS')]

def profile(row):
    return tuple((g, v) for g, v in zip(genes, row) if v != 'WT')

labels = collections.defaultdict(list)
for p, y in zip(map(profile, tr[genes].values), tr['SUBCLASS']):
    labels[p].append(y)

sub = pd.read_csv(src, encoding='utf-8-sig')
assert (sub['ID'] == te['ID']).all()
changed = collections.Counter()
for i, p in enumerate(map(profile, te[genes].values)):
    ys = labels.get(p)
    if len(p) < min_mut or not ys or len(ys) != 1 or ys[0] not in SIB:
        continue
    new = SIB[ys[0]]
    if sub.at[i, 'SUBCLASS'] != new:
        changed[f"{sub.at[i, 'SUBCLASS']}->{new}"] += 1
        sub.at[i, 'SUBCLASS'] = new

out = src.replace('.csv', '_SIB.csv')
sub.to_csv(out, index=False)
print(out, '바뀐 행', sum(changed.values()))
for k, v in changed.most_common():
    print(' ', k, v)
