"""Resolve the default JABS/balanced pair or explicitly selected recipes without importing a simulator or starting W&B."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import wandb
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
DEFAULT_KEYS = ('jabs_scratch', 'eigendexplore_scratch_bal')
KEYS = DEFAULT_KEYS + ('jabs_finetune', 'eigendexplore_finetune', 'eigendexplore_scratch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path)
    parser.add_argument('--basis', type=Path, help='Hot basis for explicitly selected hot scratch or historical eigen fine-tune')
    parser.add_argument('--balanced-basis', type=Path, help='Balanced basis for the default EigenDExplore run')
    parser.add_argument('--runs', nargs='+', choices=KEYS, default=list(DEFAULT_KEYS))
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--wandb-entity', required=True)
    parser.add_argument('--wandb-project', default='action-bench-simtoolreal')
    parser.add_argument('--wandb-group', default='isaacgym_four_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--gpus', nargs='+', type=int)
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        parser.error('--runs must not contain duplicates')
    if args.gpus is None:
        args.gpus = list(range(len(args.runs)))
    if len(args.gpus) != len(args.runs) or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error('--gpus must name one distinct nonnegative GPU per selected run')
    if args.seeds is not None and len(args.seeds) != len(args.runs):
        parser.error('--seeds must give one seed per selected run')
    paths = []
    if any(key in args.runs for key in ('eigendexplore_scratch', 'eigendexplore_finetune')):
        if args.basis is None:
            parser.error('--basis is required for an explicitly selected hot eigen recipe')
        paths.append(args.basis)
    if 'eigendexplore_scratch_bal' in args.runs:
        if args.balanced_basis is None:
            parser.error('--balanced-basis is required for the balanced run')
        paths.append(args.balanced_basis)
    for key, filename in [('jabs_finetune', 'str_joint_abs_s13.pth'),
                          ('eigendexplore_finetune', 'str_joint_abs_eignoise_espp_hot_s202.pth')]:
        if key in args.runs:
            if args.weights_dir is None:
                parser.error('--weights-dir is required for fine-tuning')
            paths.append(args.weights_dir / filename)
    for path in paths:
        if not path.is_file():
            parser.error('Required input is missing: ' + str(path))
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('--output-dir must be new; existing run IDs/configs are never overwritten')
    os.environ.update(STR_RUN_ROOT=str(args.run_root.resolve()),
                      STR_WEIGHTS_DIR=str(args.weights_dir.resolve()) if args.weights_dir else '',
                      STR_EIGEN_BASIS='',
                      STR_WANDB_ENTITY=args.wandb_entity,
                      STR_WANDB_PROJECT=args.wandb_project,
                      STR_WANDB_GROUP=args.wandb_group)
    configs, manifest = [], []
    for index, (key, gpu) in enumerate(zip(args.runs, args.gpus)):
        basis = args.balanced_basis if key == 'eigendexplore_scratch_bal' else args.basis
        os.environ['STR_EIGEN_BASIS'] = str(basis.resolve()) if basis else ''
        os.environ['STR_WANDB_ID'] = wandb.util.generate_id()
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / 'configs' / (key + '.yaml')), resolve=True)
        cfg['campaign']['gpu'] = gpu
        if args.seeds is not None:
            cfg['seed'] = args.seeds[index]
            cfg['train']['params']['seed'] = args.seeds[index]
            name = '0_gym_{}_s{}'.format(key, cfg['seed'])
            cfg.update(experiment=name, wandb_name=name, full_experiment_name=name)
            cfg['train']['params']['config'].update(name=name, full_experiment_name=name)
        run_dir = Path(cfg['campaign']['run_dir'])
        if run_dir.exists():
            parser.error('Run directory already exists: ' + str(run_dir))
        path = output / 'configs' / (key + '.yaml')
        configs.append((path, cfg))
        manifest.append(dict(cfg['campaign'], seed=cfg['seed'], config=str(path),
                             exploration_scale=cfg['train']['params']['config']['expl_reward_coef_scale'],
                             initial_learning_rate=cfg['train']['params']['config']['learning_rate'],
                             success_tolerance=cfg['task']['env']['successTolerance'],
                             wandb_url='https://wandb.ai/{}/{}/runs/{}'.format(
                                 args.wandb_entity, args.wandb_project, cfg['campaign']['wandb_id'])))
    (output / 'configs').mkdir(parents=True)
    for path, cfg in configs:
        OmegaConf.save(OmegaConf.create(cfg), path)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output / 'manifest.json')


if __name__ == '__main__':
    main()
