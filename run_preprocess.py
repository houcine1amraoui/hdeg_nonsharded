import argparse
from src.pipeline.data import config_from, prepare

def main():
    parser = argparse.ArgumentParser(description='Prepare one array per split, no shards.')
    parser.add_argument('--config', default='configs/restart.yaml')
    parser.add_argument('--allow-incomplete', action='store_true', help='Module-check data only; does not permit training without validation.')
    args = parser.parse_args()
    manifest = prepare(config_from(args.config), allow_incomplete=args.allow_incomplete)
    print({k:v['pairs'] for k,v in manifest['splits'].items()})
    print('Status:', 'incomplete; module checks only' if manifest['incomplete'] else 'all splits available')

if __name__ == '__main__':
    main()
