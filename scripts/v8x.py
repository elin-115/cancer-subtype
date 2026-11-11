# 도엽 v8A(07_v8_service_model.ipynb) 위에 우리 후보 피처를 하나씩 얹어 같은 일반 5-Fold로 비교
# 사용: python v8x.py 이름 [H] [D] [S] [N10] [WP0.7] [SUBMIT]
import sys, re, time, json, numpy as np, pandas as pd, scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
name, parts = sys.argv[1], set(sys.argv[2:])
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'blosum62.py')).read())
DATA='/Users/mac/Desktop/cancer-subtype/data'; SEED=42
HOT_N = 10 if 'N10' in parts else 5
STAGE2_C = next((float(p[1:]) for p in parts if re.fullmatch(r'C[0-9.]+', p)), 0.005)   # 2단계 규제 (예: C0.05)
S2HOT = 'S2HOT' in parts   # 2단계에 핫스팟 0/1 추가
WPOW = next((float(p[2:]) for p in parts if p.startswith('WP')), 1.0)
SIL = 'S' in parts
train=pd.read_csv(f'{DATA}/train.csv',dtype=str); test=pd.read_csv(f'{DATA}/test.csv',dtype=str)
genes=[c for c in train.columns if c not in ('ID','SUBCLASS')]; G=np.array(genes)
V=train[genes].fillna('WT').values; V_test=test.reindex(columns=genes,fill_value='WT').fillna('WT').values
le=LabelEncoder(); y=le.fit_transform(train['SUBCLASS']); CL=list(le.classes_); N=len(y)
MIS=re.compile(r'^([A-Z])(\d+)([A-Z])$')
def silent(t): m=MIS.match(t); return bool(m) and m.group(1)==m.group(3)
def blos(t):
    m=MIS.match(t)
    if not m or m.group(1)==m.group(3): return None
    return BLOSUM62.get((m.group(1),m.group(3)))
def parse_rows(V):
    hot=[]; B=np.zeros(V.shape,np.float32); T=np.zeros(V.shape,np.float32); H=np.zeros(V.shape,np.float32)
    for i,row in enumerate(V):
        h=[]
        for j in np.flatnonzero(row!='WT'):
            toks=row[j].split()
            keep=[m for m in toks if not (SIL and silent(m))]
            B[i,j]=1.0 if (keep or not SIL) else 0.0
            if any(('*' in t) or ('fs' in t) for t in toks): T[i,j]=1
            if any((b:=blos(t)) is not None and b<=-2 for t in toks): H[i,j]=1
            for m in keep:
                nums=re.findall(r'\d+',m)
                if nums: h.append(f'{G[j]}_{nums[0]}')
        hot.append(list(dict.fromkeys(h)))
    NM=(V!='WT').sum(1).astype(np.float32)   # n_mut은 항상 전체 변이 수
    # 치환 스펙트럼: missense의 원래/바뀐 아미노산 비율(20+20) + 자주 나오는 치환쌍 비율, silent의 아미노산 비율(20). 행 안에서만 계산
    AA='ACDEFGHIKLMNPQRSTVWY'; ai={a:k for k,a in enumerate(AA)}
    FR=np.zeros((len(V),20),np.float32); TO=np.zeros((len(V),20),np.float32); PR=np.zeros((len(V),400),np.float32); SL=np.zeros((len(V),20),np.float32)
    for i,row in enumerate(V):
        for c in row[row!='WT']:
            for t in c.split():
                m=MIS.match(t)
                if not m or m.group(1) not in ai or m.group(3) not in ai: continue
                f_,t_=ai[m.group(1)],ai[m.group(3)]
                if f_==t_: SL[i,f_]+=1
                else: FR[i,f_]+=1; TO[i,t_]+=1; PR[i,f_*20+t_]+=1
    nmis=FR.sum(1,keepdims=True); nsil=SL.sum(1,keepdims=True)
    SPEC=np.hstack([FR/(nmis+1),TO/(nmis+1),SL/(nsil+1)])   # +1: 변이가 적은 환자는 0 쪽으로 줄임
    PAIR=PR/(nmis+1)
    return dict(BIN=sp.csr_matrix(B),NMUT=NM,HOT=hot,T=sp.csr_matrix(T),H=sp.csr_matrix(H),SPEC=SPEC,PAIR=PAIR)
t0=time.time(); TR,TE=parse_rows(V),parse_rows(V_test); print(name,sorted(parts),'parse',f'{time.time()-t0:.0f}s',flush=True)
GRP=pd.util.hash_pandas_object(pd.DataFrame(V),index=False).values.astype(np.int64)
z=np.flatnonzero(TR['NMUT']==0); GRP[z]=-(z+1)
ident=lambda d:d
sub=lambda L,ix:[L[i] for i in ix]
def find_pairs(ix,min_pairs=20):
    labs=defaultdict(set)
    for i in ix: labs[GRP[i]].add(y[i])
    cnt=Counter(tuple(sorted(s)) for s in labs.values() if len(s)==2); tot=Counter(y[ix]); pairs={}
    for (a,b),c in cnt.items():
        if c>=min_pairs:
            spec,umb=(a,b) if tot[a]<tot[b] else (b,a); pairs[umb]=spec
    return pairs
def clean(ix,pairs):
    g=defaultdict(list)
    for i in ix: g[GRP[i]].append(i)
    keep,lab=[],[]
    for rows in g.values():
        ls={y[i] for i in rows}
        spec=[pairs[u] for u in ls if u in pairs and pairs[u] in ls] if len(ls)>1 else []
        if spec: keep.append(rows[0]); lab.append(spec[0])
        else: keep+=rows; lab+=[y[i] for i in rows]
    return np.array(keep),np.array(lab)
XGB=dict(n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.3,tree_method='hist',n_jobs=-1,random_state=SEED)
class Stage1:
    def _X(self,S,ix):
        parts_=[S['BIN'][ix],sp.csr_matrix(S['NMUT'][ix][:,None]),self.hot.transform(sub(S['HOT'],ix))]
        if 'D' in parts: parts_.append(S['T'][ix][:,self.tk])
        if 'H' in parts: parts_.append(S['H'][ix][:,self.hk])
        if 'SPEC' in parts: parts_.append(sp.csr_matrix(S['SPEC'][ix]))
        if 'PAIR' in parts: parts_.append(sp.csr_matrix(S['PAIR'][ix][:,self.pk]))
        return sp.hstack(parts_,format='csr').astype(np.float32)
    def fit(self,ix,lab):
        self.hot=CountVectorizer(analyzer=ident,min_df=HOT_N,binary=True,lowercase=False).fit(sub(TR['HOT'],ix))
        self.tk=np.flatnonzero(np.asarray(TR['T'][ix].sum(0)).ravel()>=10)
        self.hk=np.flatnonzero(np.asarray(TR['H'][ix].sum(0)).ravel()>=10)
        self.pk=np.flatnonzero((TR['PAIR'][ix]>0).sum(0)>=30)   # 학습 fold에서 30명 이상에게 나온 치환쌍만
        self.m=XGBClassifier(**XGB).fit(self._X(TR,ix),lab,sample_weight=compute_sample_weight('balanced',lab)**WPOW); return self
    def proba(self,S,ix): return self.m.predict_proba(self._X(S,ix))
class Stage2:
    def fit(self,ix,lab,pairs,hot=None):
        self.hot=hot if S2HOT else None
        self.gmask=np.flatnonzero(np.asarray(TR['BIN'][ix].sum(0)).ravel()>=5); self.models={}
        for umb,spec in pairs.items():
            m=np.isin(lab,[umb,spec])
            self.models[(umb,spec)]=LogisticRegression(C=STAGE2_C,class_weight='balanced',max_iter=2000).fit(self._X(TR,ix[m]),(lab[m]==spec).astype(int))
        return self
    def _X(self,S,ix):
        p=[S['BIN'][ix][:,self.gmask],sp.csr_matrix(np.log1p(S['NMUT'][ix])[:,None])]
        if self.hot is not None: p.append(self.hot.transform(sub(S['HOT'],ix)))
        return sp.hstack(p,format='csr')
    def predict(self,P,S,ix):
        Pm=P.copy()
        for umb,spec in self.models: Pm[:,umb]=P[:,umb]+P[:,spec]; Pm[:,spec]=-1
        pred=Pm.argmax(1)
        for (umb,spec),mdl in self.models.items():
            m=pred==umb
            if m.any(): pred[m]=np.where(mdl.predict_proba(self._X(S,ix[m]))[:,1]>=0.5,spec,umb)
        return pred
def fit_all(ix):
    pairs=find_pairs(ix); k,l=clean(ix,pairs); s1=Stage1().fit(k,l); return pairs,s1,Stage2().fit(k,l,pairs,s1.hot)
if 'GRID' in parts:   # fold마다 1단계는 한 번만 학습하고 2단계 설정만 바꿔 비교 (1단계 흔들림 제거)
    CFG=[('base C0.005',0.005,False),('C0.002',0.002,False),('C0.001',0.001,False),('C0.0003',0.0003,False),('핫스팟 C0.002',0.002,True),('핫스팟 C0.001',0.001,True)]
    O={c[0]:np.zeros(N,int) for c in CFG}; FS={c[0]:[] for c in CFG}
    for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.arange(N),y),1):
        pairs=find_pairs(tr); k,l=clean(tr,pairs); s1=Stage1().fit(k,l); P=s1.proba(TR,va)
        for nm_,c,h in CFG:
            STAGE2_C=c; S2HOT=h; s2=Stage2().fit(k,l,pairs,s1.hot); O[nm_][va]=s2.predict(P,TR,va); FS[nm_].append(f1_score(y[va],O[nm_][va],average='macro'))
        print(f'  fold {f} {time.time()-t0:.0f}s',flush=True)
    pi=[CL.index(c) for c in ['KIPAN','KIRC','GBMLGG','LGG']]
    for nm_,_,_ in CFG:
        pf=f1_score(y,O[nm_],average=None)
        print(f'{nm_:14s} CV {f1_score(y,O[nm_],average="macro"):.4f}  folds {np.round(FS[nm_],4).tolist()}  KIPAN/KIRC/GBMLGG/LGG {np.round(pf[pi],3).tolist()}',flush=True)
    sys.exit()
oof=np.zeros(N,int); fs=[]
for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.arange(N),y),1):
    pairs,s1,s2=fit_all(tr); oof[va]=s2.predict(s1.proba(TR,va),TR,va); fs.append(f1_score(y[va],oof[va],average='macro'))
    print(f'  fold {f} {fs[-1]:.4f} {time.time()-t0:.0f}s',flush=True)
cv=f1_score(y,oof,average='macro'); per=dict(zip(CL,np.round(f1_score(y,oof,average=None),3).tolist()))
print(f'RESULT {name} CV {cv:.4f}',flush=True)
import os; D=os.path.dirname(os.path.abspath(__file__))   # 결과 json·oof는 스크립트 폴더에
np.save(f'{D}/oof_{name}.npy',oof); json.dump(dict(name=name,cv=cv,folds=fs,per=per),open(f'{D}/res_{name}.json','w'))
if 'SUBMIT' in parts:
    pairs,s1,s2=fit_all(np.arange(N)); te=np.arange(len(V_test))
    s=pd.read_csv(f'{DATA}/sample_submission.csv'); s['SUBCLASS']=le.inverse_transform(s2.predict(s1.proba(TE,te),TE,te))
    out=f'/Users/mac/Desktop/cancer-subtype/output/submission_{name}.csv'; s.to_csv(out,index=False,encoding='utf-8-sig'); print('saved',out)
