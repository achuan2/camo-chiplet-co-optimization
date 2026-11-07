#!/usr/bin/env python3
"""Preregistered seed-level analysis. Never modifies raw optimizer outputs."""
import collections
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
from pathlib import Path
import zipfile
import numpy as np
import yaml
from scipy.stats import rankdata, wilcoxon, binomtest
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PLAN=yaml.safe_load((ROOT/'experiment_plan.yaml').read_text())
spec=importlib.util.spec_from_file_location('artifact_metrics',ROOT/'source/scripts/analyze_experiment_matrix.py')
util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
METRICS=PLAN['statistics']['metrics']
MAXIMIZE=set(METRICS)-{'igd_plus_5d'}
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
RNG=np.random.default_rng(20261004)

def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def write(name, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def quantile(values,q):
    a=np.sort(np.asarray(values,float));a=a[~np.isnan(a)]
    if not len(a):return math.nan
    h=(len(a)-1)*q;i=int(math.floor(h));j=int(math.ceil(h))
    if i==j or a[i]==a[j]:return float(a[i])
    if np.isinf(a[j]):return float(a[j])
    return float(a[i]+(h-i)*(a[j]-a[i]))

def bootstrap(d):
    samples=RNG.integers(0,len(d),size=(10000,len(d)))
    med=np.median(np.asarray(d)[samples],axis=1)
    return np.quantile(med,[.025,.975]).tolist()

if '--self-test' in sys.argv:
    assert quantile([1,2,3,math.inf],.5)==2.5
    assert quantile([math.inf,math.inf],.25)==math.inf
    assert quantile([0,1,2,math.inf],.75)==math.inf
    assert float(wilcoxon([1.,2.,3.],alternative='greater',zero_method='wilcox',method='exact').pvalue)==0.125
    assert float(binomtest(3,3,p=.5,alternative='greater').pvalue)==0.125
    lo,hi=bootstrap([1.,1.,1.]);assert lo==hi==1.
    assert np.array_equal(util.low_memory_nondominated_points(np.array([[1.,2.,1.,1.,1.],[2.,3.,1.,1.,1.]])),[[1.,2.,1.,1.,1.]])
    print('PASS: extended-real descriptions, exact tests, paired bootstrap and five-objective nondominance')
    sys.exit(0)

runs=[];summaries=[];scales={};fronts={}
for fam in PLAN['families']:
    completed=json.loads((ROOT/'formal'/fam['name']/'completed_runs.json').read_text())
    assert len(completed)==len(fam['tasks'])
    points={}; records={}
    for item in completed:
        t=item['task'];key=(t['method'],t['seed']);base=Path(item['output'])/'moham'
        ledger=read(base/'moham.evaluation_ledger.csv')
        final=read(base/'moham.final_population_metrics.csv')
        feasible=[r for r in ledger if util.is_feasible(r)]
        final_feasible=[r for r in final if util.is_feasible(r)]
        nd=util.common_nondominated_points(final_feasible)
        points[key]=nd
        timing=read(base/'moham.evaluation_timing.csv')
        records[key]={
          'scenario':fam['name'],'method':t['method'],'seed':t['seed'],'evaluations':len(ledger),
          'feasible_evaluations_total':len(feasible),'feasible_evaluation_rate_total':len(feasible)/len(ledger) if ledger else 0,
          'unique_feasible_designs_total':len({r['design_fingerprint'] for r in feasible}),
          'ever_feasible':int(bool(feasible)),'final_population_feasible_count':len(final_feasible),
          'final_common_nd_feasible_objective_count':len(nd),'final_empty':int(len(nd)==0),
          'first_feasible_evaluation':min((int(r['evaluation_id']) for r in feasible),default=0),
          'external_time_s':item['external_time_s'],
          'optimizer_time_s':max((float(r['completion_optimizer_seconds']) for r in timing),default=0),
          'max_admission_s':max((float(r['admission_optimizer_seconds']) for r in timing),default=0),
          'tail_seconds':max(0,max((float(r['completion_optimizer_seconds']) for r in timing),default=0)-900) if fam['timed'] else 0,
          'raw_output':item['output']}
    valid=[p for p in points.values() if len(p)]
    ref=util.low_memory_nondominated_points(np.vstack(valid)) if valid else np.empty((0,5))
    ideal=np.min(ref,axis=0) if len(ref) else np.full(5,np.nan)
    nadir=np.max(ref,axis=0) if len(ref) else np.full(5,np.nan)
    span=np.where(nadir>ideal,nadir-ideal,1)
    hv=HV(ref_point=np.full(5,1.1));igd=IGDPlus((ref-ideal)/span) if len(ref) else None
    fronts[fam['name']]=ref
    scales[fam['name']]={'objectives':util.OBJECTIVES,'ideal':ideal.tolist(),'nadir':nadir.tolist(),
                        'span':span.tolist(),'hv_reference':[1.1]*5,'reference_front_size':len(ref),
                        'construction':'union of all formal branches and seeds, unique feasible final-population five-objective ND points'}
    np.savetxt(OUT/f"{fam['name']}_reference_front.csv",ref,delimiter=',',header=','.join(util.OBJECTIVES),comments='')
    for key,p in points.items():
        row=records[key];norm=(p-ideal)/span
        row['hypervolume_5d']=float(hv(norm)) if len(p) else 0.0
        row['igd_plus_5d']=float(igd(norm)) if len(p) and igd is not None else math.inf
        runs.append(row)
    for method in fam['methods']:
        a=[r for r in records.values() if r['method']==method];assert len(a)==30
        for metric in METRICS+['feasible_evaluation_rate_total','ever_feasible','final_empty','evaluations','optimizer_time_s','external_time_s','tail_seconds']:
            values=[r[metric] for r in a];q1=quantile(values,.25);q3=quantile(values,.75)
            summaries.append({'scenario':fam['name'],'method':method,'metric':metric,'n':30,
                'median':quantile(values,.5),'q1':q1,'q3':q3,
                'iqr':q3-q1 if not (math.isinf(q1) and math.isinf(q3)) else math.nan,
                'n_nonfinite':sum(not math.isfinite(v) for v in values),'n_empty_final':sum(r['final_empty'] for r in a)})
    plt.figure(figsize=(max(5,len(fam['methods'])*1.2),4))
    plt.boxplot([[r['hypervolume_5d'] for r in records.values() if r['method']==m] for m in fam['methods']],labels=fam['methods'])
    plt.ylabel('Five-objective hypervolume');plt.title(fam['name']);plt.tight_layout()
    plt.savefig(OUT/f"{fam['name']}_hv.png",dpi=180);plt.savefig(OUT/f"{fam['name']}_hv.pdf");plt.close()

lookup={(r['scenario'],r['method'],int(r['seed'])):r for r in runs}
pairs=[];differences=[];interactions=[]
def compare(family,scenario,candidate,baseline,metric,alternative):
    seeds=next(f['seeds'] for f in PLAN['families'] if f['name']==scenario)
    x=np.asarray([lookup[scenario,candidate,s][metric] for s in seeds],float)
    y=np.asarray([lookup[scenario,baseline,s][metric] for s in seeds],float)
    orientation=1 if metric in MAXIMIZE else -1
    finite=np.isfinite(x)&np.isfinite(y)
    d=orientation*(x[finite]-y[finite]);nz=d[d!=0]
    ties=len(nz)-len(np.unique(np.abs(nz)))
    test='NA';p=math.nan;rb=math.nan
    if len(d):
        if not len(nz):p=1.0;rb=0.0;test='all-zero'
        else:
            test='exact' if len(nz)==len(d) and ties==0 else 'approx'
            p=float(wilcoxon(d,alternative=alternative,zero_method='wilcox',method=test).pvalue)
            ranks=rankdata(np.abs(nz));rb=float((ranks[nz>0].sum()-ranks[nz<0].sum())/ranks.sum())
    row={'family':family,'scenario':scenario,'candidate':candidate,'baseline':baseline,'metric':metric,
         'alternative':alternative,'n_pairs':int(finite.sum()),'n_zero_differences':int(sum(d==0)),
         'absolute_difference_ties':ties,'test_method':test,'median_oriented_paired_difference':quantile(d,.5),
         'rank_biserial':rb,'p_raw':p,'p_holm':math.nan,'wins':int(sum(d>0)),'losses':int(sum(d<0))}
    if metric=='hypervolume_5d':row['bootstrap_ci_low'],row['bootstrap_ci_high']=bootstrap(d)
    pairs.append(row)
    for seed,a,b in zip(seeds,x,y):
        if a==b: extended=0
        else: extended=(1 if a>b else -1)*orientation
        differences.append({'family':family,'scenario':scenario,'candidate':candidate,'baseline':baseline,'metric':metric,
            'seed':seed,'candidate_value':a,'baseline_value':b,'oriented_difference':orientation*(a-b) if math.isfinite(a) and math.isfinite(b) else math.nan,
            'extended_real_sign':extended})
    if metric=='igd_plus_5d':
        signs=[(0 if a==b else (1 if a>b else -1)*orientation) for a,b in zip(x,y)]
        wins=sum(s>0 for s in signs);losses=sum(s<0 for s in signs)
        signp=float(binomtest(wins,wins+losses,p=.5,alternative=alternative).pvalue) if wins+losses else 1.0
        pairs.append({'family':family+'-sign','scenario':scenario,'candidate':candidate,'baseline':baseline,'metric':metric,
                      'alternative':alternative,'n_pairs':30,'n_zero_differences':30-wins-losses,
                      'test_method':'exact-extended-real-sign','wins':wins,'losses':losses,'p_raw':signp,'p_holm':math.nan})

for scenario in ['normal','slo']:
    for method in ['V10','V01','V11']:
        for metric in METRICS:compare('A',scenario,method,'V00',metric,'two-sided')
    fam=next(f for f in PLAN['families'] if f['name']==scenario)
    d=[]
    for s in fam['seeds']:
        value=lookup[scenario,'V11',s]['hypervolume_5d']-lookup[scenario,'V10',s]['hypervolume_5d']-lookup[scenario,'V01',s]['hypervolume_5d']+lookup[scenario,'V00',s]['hypervolume_5d']
        d.append(value);interactions.append({'scenario':scenario,'seed':s,'interaction':value})
    lo,hi=bootstrap(d)
    interactions.append({'scenario':scenario,'seed':'paired-seed-block-summary','interaction':float(np.median(d)),'ci_low':lo,'ci_high':hi})
for fam in PLAN['families']:
    for metric in METRICS:compare('B',fam['name'],'CAMO','V11',metric,'greater')

families=collections.defaultdict(list)
for row in pairs:families[row['family'],row['metric']].append(row)
for key,group in families.items():
    assert len(group)==6,(key,len(group))
    ranked=sorted(group,key=lambda r:r['p_raw'] if math.isfinite(r['p_raw']) else 1.0)
    running=0.0
    for i,row in enumerate(ranked):
        raw=row['p_raw'] if math.isfinite(row['p_raw']) else 1.0
        running=max(running,min(1.,raw*(6-i)))
        row['p_holm']=running if math.isfinite(row['p_raw']) else math.nan
        row['holm_family_size']=6
write('run_metrics.csv',runs);write('summary_metrics.csv',summaries);write('paired_tests.csv',pairs)
write('paired_seed_differences.csv',differences);write('hv_interactions.csv',interactions)
manifest={'plan_sha256':sha(ROOT/'experiment_plan.yaml'),'analysis_script_sha256':sha(__file__),
          'normalization':scales,'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pymoo','matplotlib','PyYAML']},
          'bootstrap_seed':20261004,'bootstrap_samples':10000,'bootstrap_statistic':'median oriented paired HV difference / median four-way interaction',
          'bootstrap_scale':'fixed original empirical reference scale; no refit',
          'quantiles':'linear order-statistic interpolation extended to +inf; inf-inf IQR undefined (NA)',
          'zero_feasible':'HV=0 IGD+=+inf; all 30 retained for description, finite pairs for Wilcoxon; sign uses extended-real ordering',
          'formal_runs':len(runs),'holm_families':{f'{k[0]}:{k[1]}':6 for k in families},
          'raw_results_mutated':False}
(OUT/'analysis_manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
summary=['# CAMO supplementary experiment handoff','',f'Validated formal runs: {len(runs)}/660.',
 'Two C-TAEA switches are sensitivity factors, not a claim of complete C-TAEA reproduction.',
 'The deadline control binary is new and common to all branches. Historical results are unchanged.','',
 '## Confirmatory CAMO versus V11 hypervolume comparisons','']
for row in pairs:
    if row['family']=='B' and row['metric']=='hypervolume_5d':
        outcome='positive oriented median' if row['median_oriented_paired_difference']>0 else 'non-positive oriented median'
        summary.append(f"- {row['scenario']}: {outcome}; paired median difference={row['median_oriented_paired_difference']:.8g}, 95% CI=[{row['bootstrap_ci_low']:.8g}, {row['bootstrap_ci_high']:.8g}], Holm p={row['p_holm']:.8g}, rank-biserial={row['rank_biserial']:.8g}.")
summary+=['','## Interpretation boundaries','',
 '不显著不等于等效；不得跨设置直接比较绝对HV。全部逐种子结果、空集和失败记录均保留。',
 '旧C-TAEA二进制未恢复的证据缺口仍存在。本轮闭合的是新源码、构建和运行的版本链。',
 '稿件核对位置：Comparison Methods、版本说明、RQ1/RQ3/RQ4、Supplement、图2。不得自动替换旧正式表格。','',
 '## English Methods draft','',
 'We conducted a preregistered paired sensitivity study of two C-TAEA mating rules using four combinations on Normal and SLO-tight workloads, and combined-rule controls on the remaining settings. All branches shared one versioned optimizer and workload-specific frozen configuration. Fixed-evaluation runs used 3,002 true evaluations; fixed-time runs used a common 900-second admission deadline with in-flight evaluations allowed to complete. Thirty paired seeds were retained in each setting. Empirical normalization and reference fronts were shared within, but not across, settings. Holm families were separated between rule sensitivity and CAMO-versus-V11 comparisons.','',
 '## Results / Supplement draft','',
 'Use the six numerical comparisons above together with summary_metrics.csv and paired_tests.csv. Report reductions, nonsignificant comparisons, reversals, and empty sets explicitly. The four-factor interaction estimates are secondary paired estimates, not additional confirmatory significance tests.','']
(ROOT/'handoff/CAMO_CTAEA_SUPPLEMENT_HANDOFF.md').write_text('\n'.join(summary))
checksum=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '.git' in p.parts or 'build' in p.parts or p.name.endswith(('.zip','.tmp')) or p.name=='SHA256SUMS.txt':continue
    checksum.append(f'{sha(p)}  {p.relative_to(ROOT)}')
for name in ['moham','compile_commands.json']:
    p=ROOT/'source/build'/name
    if p.exists():checksum.append(f'{sha(p)}  {p.relative_to(ROOT)}')
(ROOT/'handoff/SHA256SUMS.txt').write_text('\n'.join(checksum)+'\n')
# Complete raw outputs stay on Ubuntu; portable bundle carries evidence and tables, not GB-scale ledgers.
with zipfile.ZipFile(ROOT/'handoff/CAMO_CTAEA_SUPPLEMENT_HANDOFF.zip','w',zipfile.ZIP_DEFLATED) as z:
    for area in ['refs','analysis','handoff','source/src','source/tests','source/scripts','source/docker']:
        for p in (ROOT/area).rglob('*'):
            if p.is_file() and not p.name.endswith('.zip') and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
    for name in ['experiment_plan.yaml','plan_lock.json','source_assets.json','supplement.py','analyze_supplement.py','build.log']:
        z.write(ROOT/name,name)
    z.write(ROOT/'source/build/moham','source/build/moham')
