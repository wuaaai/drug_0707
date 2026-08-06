"""消融实验：串行训练 6 个变体，量化各创新独立贡献。"""
import subprocess, os, sys, time

BASE = [sys.executable, '-u', 'main.py', '--model', 'TrajectoryCare',
        '--dataset', 'mimic3', '--task', 'drug_rec_ts',
        '--epochs', '30', '--test_epochs', '1', '--batch_size', '32',
        '--lr', '2e-4', '--dropout', '0.3', '--no-scheduler']

# (名称, 附加参数, 环境变量)
VARIANTS = [
    ('full',       ['--use_hetero_kg', '--use_drug_cooccurrence', '--use_lab_encoder'], {}),
    ('noLab',      ['--use_hetero_kg', '--use_drug_cooccurrence'], {}),
    ('noRoute',    ['--use_hetero_kg', '--use_drug_cooccurrence', '--use_lab_encoder'], {'ROUTING': '0'}),
    ('noJaccard',  ['--use_hetero_kg', '--use_drug_cooccurrence', '--use_lab_encoder'], {'JACCARD_W': '0'}),
    ('noCooccur',  ['--use_hetero_kg', '--use_lab_encoder'], {}),
    ('noContrast', ['--use_hetero_kg', '--use_drug_cooccurrence', '--use_lab_encoder'], {'CONTRASTIVE_W': '0'}),
]

log_dir = 'logs/20260804'
os.makedirs(log_dir, exist_ok=True)

start_all = time.time()
for name, args, env_extra in VARIANTS:
    t0 = time.time()
    cmd = BASE + args + ['--notes', f'消融_{name}']
    e = os.environ.copy()
    e.update(env_extra)
    log = os.path.join(log_dir, f'ablation_{name}.log')
    print(f'=== [{name}] 开始: {" ".join(cmd[-2:])} ===', flush=True)
    with open(log, 'w', encoding='utf-8') as f:
        r = subprocess.run(cmd, env=e, stdout=f, stderr=subprocess.STDOUT)
    print(f'=== [{name}] 完成 exit={r.returncode}, 耗时 {time.time()-t0:.0f}s ===', flush=True)

print(f'\n全部消融完成, 总耗时 {time.time()-start_all:.0f}s', flush=True)
