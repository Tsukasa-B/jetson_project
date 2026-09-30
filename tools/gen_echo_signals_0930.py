import numpy as np, pandas as pd
DT=0.02; F_HOLD=0.10; G_HOLD=0.30
def write(name, segs):
    df=[];tag=[]
    for v,t in segs: df+=list(v); tag+=[t]*len(v)
    d=pd.DataFrame({'cmd_pressure_DF':np.clip(df,0,0.6)})
    d['cmd_pressure_F']=F_HOLD; d['cmd_pressure_G']=G_HOLD
    d.insert(0,'time',np.arange(len(d))*DT)
    d.to_csv(f'test_signals/{name}.csv',index=False)
    d.assign(segment=tag).to_csv(f'test_signals/{name}_annotated.csv',index=False)
    print(f'{name}: {len(d)*DT/60:.1f} 分')
def const(v,sec): return np.full(int(round(sec/DT)),v)
# A: 正弦（中心×振幅×周波数）
rng=np.random.default_rng(1); segs=[(const(0.1,2),'rest')]
conds=[(c,a,f) for c in (0.15,0.30,0.45) for a in (0.05,0.10,0.15) for f in (0.5,1,2,3,4,6,8)]
for rep in range(2):
    for i in rng.permutation(len(conds)):
        c,a,f=conds[i]; n=int(4/DT); t=np.arange(n)*DT
        w=np.minimum(1,np.minimum(t,t[::-1])/0.5)
        segs+=[(const(c,1.0),f'settle_c{c}'),(c+a*w*np.sin(2*np.pi*f*t),f'sine_c{c}_a{a}_f{f}_r{rep}')]
segs.append((const(0.1,2),'rest')); write('echo_sine',segs)
# B: 高圧側・大ステップ
segs=[(const(0.1,2),'rest')]
for rep in range(3):
    for tg in (0.3,0.4,0.5,0.55,0.6):
        segs+=[(const(0.1,1.5),'pre'),(const(tg,3.0),f'up_0.1to{tg}_r{rep}')]
    for tg in (0.0,0.1,0.3):
        segs+=[(const(0.6,3.0),'pre'),(const(tg,2.0),f'down_0.6to{tg}_r{rep}')]
segs.append((const(0.1,2),'rest')); write('echo_bigstep',segs)
