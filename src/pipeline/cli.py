import argparse
from src.pipeline.data import config_from, SPLITS


def stage_main(stage):
    parser = argparse.ArgumentParser(description=f'Non-sharded {stage.upper()} execution.')
    parser.add_argument('--config',default='configs/restart.yaml')
    parser.add_argument('--split',choices=SPLITS,default='train')
    parser.add_argument('--run',default='module_checks')
    parser.add_argument('--checkpoint',help='Default: shared initialization checkpoint.')
    parser.add_argument('--initialize',action='store_true',help='DBRL only: create shared untrained checkpoint.')
    parser.add_argument('--require-trained',action='store_true')
    args = parser.parse_args()
    from src.pipeline.runtime import export_stage
    export_stage(config_from(args.config),args,stage)


def training_main():
    parser = argparse.ArgumentParser(description='Live non-sharded HDEG training; train/val only.')
    parser.add_argument('--config',default='configs/restart.yaml')
    parser.add_argument('--run',default='e2e_001')
    parser.add_argument('--initialization',help='Default: shared initialization checkpoint from DBRL.')
    args = parser.parse_args()
    from src.pipeline.runtime import train
    train(config_from(args.config),args)
