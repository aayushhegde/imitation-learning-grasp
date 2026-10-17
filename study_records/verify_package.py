"""Verify this package's bytes and summarize saved original outcomes; no simulation."""
from pathlib import Path
import hashlib
import json
from collections import Counter

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'PACKAGE_SHA256.json').read_text())
failures = []
for item in manifest['files']:
    p = root / item['path']
    if not p.is_file() or p.stat().st_size != item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
        failures.append(item['path'])
if failures:
    raise SystemExit('FAILED checksum verification: ' + ', '.join(failures))
print(f"PASS: {len(manifest['files'])} file checksums")
data = root / 'imitation-learning-grasp/artifacts/revision_study'
for p in sorted(data.glob('*_eval_40000_100.json')):
    rows = json.loads(p.read_text())
    assert len(rows) == 100
    assert sorted(r['seed'] for r in rows) == list(range(40000,40100))
    traces = data / p.name.replace('_eval_', '_traces_').removesuffix('.json')
    assert all((traces/f'{r["seed"]}.npz').is_file() for r in rows)
    print(p.name, 'strict successes', sum(r['strict_success'] for r in rows), '/ 100')
validation = json.loads((data / 'oracle_validation.json').read_text())
for condition in sorted({r['condition'] for r in validation}):
    rows = [r for r in validation if r['condition'] == condition]
    assert len(rows) == 30
    print('validation', condition, sum(r['strict_success'] for r in rows), '/ 30')
history = [r for r in validation if r['condition'] == 'expert_history']
print('Original expert-history phase matches:', sum(r['phase_matches'] for r in history), '/', sum(r['queries'] for r in history))
a = {r['seed']:r for r in json.loads((data/'gru33_floor_s0_eval_40000_100.json').read_text())}
b = {r['seed']:r for r in json.loads((data/'dagger_gru33_floor_s0_eval_40000_100.json').read_text())}
assert set(a) == set(b)
paired = Counter((a[s]['strict_success'],b[s]['strict_success']) for s in a)
print('Paired baseline/DAgger strict successes:',sum(v['strict_success'] for v in a.values()),sum(v['strict_success'] for v in b.values()))
print('Improved:',paired[(False, True)],'Regressed:',paired[(True, False)])
collection = json.loads((data/'dagger_round1_episodes.json').read_text())
assert sorted(r['seed'] for r in collection) == list(range(1000,1100))
assert sum(r['row_stop']-r['row_start'] for r in collection) == 7218
print('Collection: 100 histories, 7218 labels;',sum(r['strict_success'] for r in collection),'mixed-policy successes;',sum(r['abstention'] is not None for r in collection),'abstentions')
