"""New 2026-10-16 diagnostic replay; does not replace historical validation logs.

The four-condition control protocol exactly follows run_revision_dagger.validate.
Additional memory snapshots/comparisons are read-only and never control actions.
"""
from __future__ import annotations
import csv
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent.parent / 'imitation-learning-grasp', help='Repository root; package-relative default')
parser.add_argument('--output', type=Path, required=True, help='New diagnostic output directory; must not contain diagnostic_manifest.json')
args = parser.parse_args()
ROOT = args.repo.expanduser().resolve()
OUT = args.output.expanduser().resolve()
if OUT == (ROOT / 'artifacts/revision_study').resolve():
    parser.error('Refusing to write into the historical revision_study output directory')
if (OUT / 'diagnostic_manifest.json').exists():
    parser.error('Refusing to overwrite an existing diagnostic run; choose a new --output directory')
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from ilgrasp.experts import HistoryAwareRecoveryOracle, VerifiedExpertPhase, VerifiedPickPlaceExpert
from ilgrasp.experts.verified import VerifiedExpertConfig
from ilgrasp.sim import VerifiedPlacementEnv
from ilgrasp.sim.verified_task import VerifiedTaskConfig

FIELDS = ('phase', 'phase_steps', 'grasp_offset', 'grasp_target', 'release_target')
ATOL, RTOL = 1e-8, 0.0

def default(v):
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    raise TypeError(type(v).__name__)

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=default)+'\n')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot(controller):
    return {k: (controller.phase.value if k == 'phase' else
                getattr(controller, k).copy().tolist() if isinstance(getattr(controller, k), np.ndarray)
                else getattr(controller, k)) for k in FIELDS} | {
        k: getattr(controller, k) for k in ('close_steps', 'open_steps', '_step') if hasattr(controller, k)}

def compare(a, b):
    out = {}
    for key in FIELDS:
        x, y = a[key], b[key]
        if x is None or y is None:
            same, delta = x is None and y is None, None
        elif isinstance(x, str):
            same, delta = x == y, None
        else:
            same = bool(np.allclose(x, y, atol=ATOL, rtol=RTOL))
            delta = float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        out[key] = {'equal': same, 'max_abs_difference': delta}
    return out

history_path = ROOT / 'artifacts/revision_study/oracle_validation.json'
history = {(r['condition'], r['seed']): r for r in json.loads(history_path.read_text())}
historical_manifest = json.loads((ROOT / 'artifacts/revision_study/oracle_validation_manifest.json').read_text())
study_manifest = json.loads((ROOT / 'artifacts/revision_study/manifest_40000_100.json').read_text())
sources = ['scripts/run_revision_dagger.py', 'tests/test_recovery_oracle.py',
           'src/ilgrasp/experts/recovery.py', 'src/ilgrasp/experts/verified.py',
           'src/ilgrasp/sim/verified_task.py', 'src/ilgrasp/sim/observation.py',
           'src/ilgrasp/sim/env.py']
configs = {'expert': asdict(VerifiedExpertConfig()), 'task': asdict(VerifiedTaskConfig())}
manifest = {
    'diagnostic_date': '2026-10-16', 'started_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'running', 'provenance': 'New diagnostic replay, not an original validation log',
    'source_root': str(ROOT), 'source_sha256': {s: sha(ROOT / s) for s in sources},
    'diagnostic_script_sha256': sha(Path(__file__)), 'historical_results_sha256': sha(history_path),
    'historical_validation_manifest_sha256': sha(ROOT / 'artifacts/revision_study/oracle_validation_manifest.json'),
    'historical_oracle_sha256': historical_manifest['oracle_sha256'],
    'oracle_source_matches_historical_hash': sha(ROOT / 'src/ilgrasp/experts/recovery.py') == historical_manifest['oracle_sha256'],
    'other_historical_source_hashes': 'Not recorded in original validation manifest; cannot hash-verify those against its execution',
    'configs': configs,
    'config_sha256': {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k,v in configs.items()},
    'historical_config_hash_comparison': 'Original validation manifest has no serialized controller/task configs or hashes',
    'versions': {k: metadata.version(k) for k in ('torch','numpy','panda-gym','gymnasium','pybullet')},
    'historical_study_versions': study_manifest['versions'], 'python': sys.version, 'platform': platform.platform(),
    'memory_comparison': {'fields': FIELDS, 'atol': ATOL, 'rtol': RTOL, 'none_rule': 'Equal iff both are None'},
    'step_convention': {'pre_step': '0-based executed-action count before query',
                        'executed_step': '1-based action number, equal to post-step environment step'},
    'reset_mode': 'independent', 'seeds': list(range(2000,2030)),
}
manifest['versions_match_historical_study'] = manifest['versions'] == manifest['historical_study_versions']
write(OUT/'diagnostic_manifest.json', manifest)
records=[]
onsets=[]
mem_counts = {point:{k:Counter() for k in FIELDS} for point in ('pre_act','post_observe')}
first_diffs = {point:{} for point in ('pre_act','post_observe')}
comparison_diffs=[]
control_fields = ['strict_success','native_success','barrier_violation','collision_free','task_stage',
                  'step','queries','phase_matches','interventions','abstention']
with (OUT/'step_records.jsonl').open('w') as trace:
    for condition in ('oracle','transport_perturbation','open_during_close','expert_history'):
        with VerifiedPlacementEnv(reset_mode='independent') as env:
            for seed in range(2000,2030):
                env.reset(seed=seed)
                oracle = HistoryAwareRecoveryOracle(env)
                expert = VerifiedPickPlaceExpert()
                interventions=0; abstention=None; queries=0; phase_matches=0
                hit_steps=[]
                for _ in range(env.config.max_episode_steps):
                    pre_step = int(env.last_info['step'])
                    pre_oracle, pre_expert = snapshot(oracle), snapshot(expert)
                    event = {'condition':condition,'seed':seed,'pre_step':pre_step,
                             'pre_oracle':pre_oracle,'pre_expert':pre_expert,
                             'pre_observation':np.asarray(env.raw_obs['observation']).tolist(),
                             'pre_goal':np.asarray(env.raw_obs['desired_goal']).tolist()}
                    label=oracle.query(); queries+=1
                    event['label_phase']=label.phase
                    event['label_valid']=label.valid
                    event['label_reason']=label.reason
                    if not label.valid:
                        abstention=label.reason
                        event.update({'executed_step':None,'perturbation':False})
                        trace.write(json.dumps(event,allow_nan=False,default=default)+'\n')
                        break
                    action=label.action
                    event['proposed_action']=label.action.copy().tolist()
                    is_perturbation=False
                    if condition=='transport_perturbation' and interventions<3:
                        if oracle.phase is VerifiedExpertPhase.CROSS_BARRIER:
                            action=np.array([0.3,-0.2,0.0,-1.0],dtype=np.float32)
                            interventions+=1; is_perturbation=True
                    elif condition=='open_during_close' and interventions<3:
                        if oracle.phase is VerifiedExpertPhase.CLOSE:
                            action=np.array([0.0,0.0,0.0,1.0],dtype=np.float32)
                            interventions+=1; is_perturbation=True
                    elif condition=='expert_history':
                        phase_matches+=int(label.phase==expert.phase.value)
                        event['pre_act_comparison']=compare(pre_oracle,pre_expert)
                        action=expert.act(env.raw_obs)
                        event['oracle_action_matches_expert_action']=bool(np.allclose(label.action,action,atol=ATOL,rtol=RTOL))
                        event['oracle_action_max_abs_difference']=float(np.max(np.abs(label.action-action)))
                    _,_,done,truncated,_=env.step(action)
                    oracle.observe_transition()
                    post_oracle,post_expert=snapshot(oracle),snapshot(expert)
                    event.update({'executed_step':int(env.last_info['step']),
                                  'actual_action':env.last_action.copy().tolist(),
                                  'perturbation':is_perturbation,'intervention_count':interventions,
                                  'post_oracle':post_oracle,'post_expert':post_expert,
                                  'post_observation':np.asarray(env.raw_obs['observation']).tolist(),
                                  'done':bool(done),'truncated':bool(truncated)})
                    assert event['executed_step']==pre_step+1
                    if is_perturbation:hit_steps.append({'pre_step':pre_step,'executed_step':event['executed_step']})
                    if condition=='expert_history':
                        event['post_observe_comparison']=compare(post_oracle,post_expert)
                        for point in ('pre_act','post_observe'):
                            for field,val in event[point+'_comparison'].items():
                                mem_counts[point][field]['equal' if val['equal'] else 'different']+=1
                                if not val['equal'] and field not in first_diffs[point]:
                                    first_diffs[point][field]={'seed':seed,'pre_step':pre_step,
                                        'executed_step':event['executed_step'],
                                        'oracle':(pre_oracle if point=='pre_act' else post_oracle)[field],
                                        'expert':(pre_expert if point=='pre_act' else post_expert)[field],
                                        **val}
                    trace.write(json.dumps(event,allow_nan=False,default=default)+'\n')
                    if done or truncated:break
                record={'condition':condition,'seed':seed,'interventions':interventions,
                        'queries':queries,'phase_matches':phase_matches,'abstention':abstention,**env.last_info}
                records.append(record)
                historical=history[(condition,seed)]
                diffs={k:{'historical':historical[k],'diagnostic':record[k]} for k in control_fields if historical[k]!=record[k]}
                numeric_diffs={k:{'historical':historical[k],'diagnostic':record[k]}
                               for k in historical if isinstance(historical[k],float) and historical[k]!=record[k]}
                if diffs or numeric_diffs:comparison_diffs.append({'condition':condition,'seed':seed,'control_differences':diffs,'float_exact_differences':numeric_diffs})
                onsets.append({'condition':condition,'seed':seed,'first_pre_step':hit_steps[0]['pre_step'] if hit_steps else None,
                               'first_executed_step':hit_steps[0]['executed_step'] if hit_steps else None,
                               'intervention_pre_steps':[x['pre_step'] for x in hit_steps],
                               'intervention_executed_steps':[x['executed_step'] for x in hit_steps],
                               'episode_steps':record['step'],'interventions':interventions})
                if (seed-1999)%10==0:
                    print(json.dumps({'progress_condition':condition,'completed_seeds':seed-1999}),flush=True)
        write(OUT/'diagnostic_episode_results.json',records)
        trace.flush()
        print(json.dumps({'completed_condition':condition,'episodes':30,
              'strict_successes':sum(r['strict_success'] for r in records if r['condition']==condition)}),flush=True)
write(OUT/'intervention_steps.json',onsets)
with (OUT/'intervention_steps.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(onsets[0]));w.writeheader()
    w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in onsets)
summary={'diagnostic_date':'2026-10-16','historical_comparison_fields':control_fields,
         'historical_comparison':{'episodes':len(records),'all_control_fields_match':not any(r['control_differences'] for r in comparison_diffs),
                                  'all_original_float_fields_exact_match':not any(r['float_exact_differences'] for r in comparison_diffs),
                                  'differences':comparison_diffs},
         'memory_counts':mem_counts,'first_memory_differences':first_diffs,
         'condition_summary':{},'onset_ranges':{}}
for condition in ('oracle','transport_perturbation','open_during_close','expert_history'):
    rs=[r for r in records if r['condition']==condition]
    summary['condition_summary'][condition]={'episodes':len(rs),'strict_successes':sum(r['strict_success'] for r in rs),
        'abstentions':sum(r['abstention'] is not None for r in rs),'queries':sum(r['queries'] for r in rs),
        'phase_matches':sum(r['phase_matches'] for r in rs),'intervention_counts':dict(Counter(r['interventions'] for r in rs))}
    starts=[r['first_executed_step'] for r in onsets if r['condition']==condition and r['first_executed_step'] is not None]
    if starts:summary['onset_ranges'][condition]={'minimum_first_executed_step':min(starts),'maximum_first_executed_step':max(starts),
                                               'first_executed_step_counts':dict(Counter(starts))}
write(OUT/'diagnostic_summary.json',summary)
manifest['status']='complete';manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
manifest['output_sha256']={n:sha(OUT/n) for n in ('step_records.jsonl','diagnostic_episode_results.json','intervention_steps.json','intervention_steps.csv','diagnostic_summary.json')}
write(OUT/'diagnostic_manifest.json',manifest)
print(json.dumps({'completed':True,'historical_comparison':summary['historical_comparison'],
                  'onset_ranges':summary['onset_ranges'],'memory_counts':mem_counts,'first_memory_differences':first_diffs},default=default),flush=True)
