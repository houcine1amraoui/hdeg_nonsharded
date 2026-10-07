"""Shared checkpoint, non-sharded stage execution and end-to-end training."""
import json
import os
import platform
import random
import sys
from pathlib import Path
import numpy as np
import torch
from src.common.graph.semantics import load_behavioral_state_config
from src.models.hdeg.mbai import MultiScaleBehavioralAnomalyInference
from src.pipeline.model import build_live_model
from src.pipeline.data import digest, load_dataset, resolve, split_array, window_batches, write_json

LEVELS = ('Z','S','S_tilde','g')
STAGES = {'dbrl':'Z', 'bse':'S', 'bil':'S_tilde', 'ebrl':'g'}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def context(config):
    root, manifest = load_dataset(config)
    device = torch.device(config['training']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    semantic = load_behavioral_state_config(
        config_path=resolve(config['data']['semantic_config']), devices_path=root/'devices.json')
    seed_everything(int(config['seed']))
    mask = semantic.torch_mask(dtype=torch.float32, device=device, clone=True)
    model = build_live_model(config=config, behavioral_config=semantic,
                             compatibility_mask=mask, device=device)
    return root, manifest, model, device


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def checkpoint_payload(model, config, root, status, **extra):
    return {'schema_version':1, 'status':status, 'model_state_dict':model.state_dict(),
            'config':config, 'dataset_sha256':digest(root/'dataset.json'),
            'environment':{'python':sys.version, 'torch':torch.__version__,
                           'numpy':np.__version__, 'platform':platform.platform()},
            'command':sys.argv,
            'source_sha256':{str(p.relative_to(resolve('.'))):digest(p)
                for p in sorted(resolve('src').rglob('*.py'))}, **extra}


def restore(path, model, config, root):
    path = resolve(path)
    # Load only checkpoints produced by this project or otherwise trusted by the user.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['dataset_sha256'] != digest(root/'dataset.json'):
        raise ValueError('Checkpoint and preprocessed dataset identities differ.')
    if payload['config']['hdeg'] != config['hdeg']:
        raise ValueError('Checkpoint and requested model configuration differ.')
    if int(payload['config']['seed']) != int(config['seed']):
        raise ValueError('Checkpoint and configured initialization seed differ.')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return payload


def export_stage(config, args, stage):
    root, manifest, model, device = context(config)
    checkpoint = resolve(args.checkpoint) if args.checkpoint else root/'checkpoints/initialization.pt'
    if args.initialize:
        if stage != 'dbrl' or args.checkpoint:
            raise ValueError('--initialize is only valid for DBRL without --checkpoint.')
        if checkpoint.exists():
            raise FileExistsError(f'{checkpoint} already exists. Reuse it without --initialize.')
        atomic_checkpoint(checkpoint, checkpoint_payload(model,config,root,'untrained_initialization',epoch=0))
    payload = restore(checkpoint, model, config, root)
    if args.require_trained and payload['status'] != 'trained_selected':
        raise ValueError('A selected trained checkpoint is required.')
    if not args.run or Path(args.run).name != args.run or args.run in ('.','..'):
        raise ValueError('--run must be a single directory name.')
    folder = root/'exports'/args.run/args.split
    destination = folder/stage
    if destination.exists():
        raise FileExistsError(f'{destination} exists; use a new --run name.')
    identity = {'checkpoint_sha256':digest(checkpoint), 'dataset_sha256':digest(root/'dataset.json'),
                'checkpoint_status':payload['status'], 'split':args.split}
    def load_stage(name):
        directory = folder/name
        meta = json.loads((directory/'stage.json').read_text())
        if any(meta[k] != v for k,v in identity.items()):
            raise ValueError(f'{name}: checkpoint/dataset/split mismatch.')
        arrays = {}
        for key, description in meta['arrays'].items():
            path = directory/f'{key}.npy'
            if digest(path) != description['sha256']:
                raise ValueError(f'{name}/{key}: artifact hash mismatch.')
            arrays[key] = np.load(path, mmap_mode='r', allow_pickle=False)
        return arrays
    batch_size = int(config['training']['batch_size'])
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    model.eval()
    tensor = lambda x: torch.tensor(np.asarray(x),dtype=torch.float32,device=device)
    if stage == 'dbrl':
        raw = split_array(root, manifest, args.split)
        count = max(0,len(raw)-manifest['window_size']+1)
        batches = ((start, {'Z':model.dbrl(tensor(x))}) for start,x,_ in
                   window_batches(raw, manifest['window_size'],batch_size,paired=False))
    elif stage in ('bse','bil','ebrl'):
        upstream, key = {'bse':('dbrl','Z'), 'bil':('bse','S'), 'ebrl':('bil','S_tilde')}[stage]
        array = load_stage(upstream)[key]
        count = len(array)
        def encode():
            for start in range(0,count,batch_size):
                x = tensor(array[start:start+batch_size])
                out = model.bse(x,model.compatibility_mask) if stage=='bse' else getattr(model,stage)(x)
                yield start, {STAGES[stage]:out}
        batches = encode()
    else:
        observed = {key:load_stage(name)[key] for name,key in STAGES.items()}
        lengths = {len(a) for a in observed.values()}
        if len(lengths) != 1:
            raise ValueError('Representation length mismatch.')
        count = max(0,len(observed['g'])-1)
        predicted = load_stage('hbf') if stage in ('mo','mbai') else None
        if predicted and any(len(a)!=count for a in predicted.values()):
            raise ValueError('Forecast and next-window target counts differ.')
        mbai = MultiScaleBehavioralAnomalyInference().to(device)
        def predict_or_score():
            for start in range(0,count,batch_size):
                stop = min(count,start+batch_size)
                if stage=='hbf':
                    yield start, model.forecast({k:tensor(v[start:stop]) for k,v in observed.items()})
                else:
                    future = {k:tensor(v[start+1:stop+1]) for k,v in observed.items()}
                    forecast = {k:tensor(v[start:stop]) for k,v in predicted.items()}
                    if stage=='mbai':
                        yield start, mbai(future,forecast)
                    else:
                        # Store per-pair objective values; reduction matches ModelOptimization.
                        losses = {k:(future[k]-forecast[k]).square().flatten(1).mean(1) for k in LEVELS}
                        total = sum(getattr(model.mo,weight)*losses[k] for k,weight in
                                    zip(LEVELS,('lambda_Z','lambda_S','lambda_S_tilde','lambda_G')))
                        yield start, {**{f'L_{k}':v for k,v in losses.items()}, 'L_HDEG':total}
        batches = predict_or_score()
    if count <= 0:
        raise ValueError(f'{args.split}: insufficient observations for {stage}.')
    destination.mkdir(parents=True)
    buffers = {}
    with torch.no_grad():
        for start, outputs in batches:
            for key, value in outputs.items():
                array = value.detach().cpu().numpy()
                if not np.isfinite(array).all():
                    raise ValueError(f'{stage}/{key}: nonfinite output.')
                if key not in buffers:
                    buffers[key] = np.lib.format.open_memmap(destination/f'{key}.npy', mode='w+',
                                      dtype=array.dtype, shape=(count,*array.shape[1:]))
                buffers[key][start:start+len(array)] = array
    for array in buffers.values():
        array.flush()
    arrays = {key:{'shape':list(array.shape),'sha256':digest(destination/f'{key}.npy')}
              for key,array in buffers.items()}
    write_json(destination/'stage.json', {**identity, 'stage':stage,'arrays':arrays,
        'count':count,'command':sys.argv,
        'purpose':'scientific_artifact_candidate' if payload['status']=='trained_selected' else 'module_check_only',
        'alignment':'row i predicts/compares observed row i+1' if stage in ('hbf','mo','mbai') else 'row i encodes raw[i:i+W]'})
    print(f'{stage}: {count} rows; checkpoint status={payload["status"]}; {destination}')


def epoch_pass(model, array, w, batch_size, device, optimizer=None):
    model.train(optimizer is not None)
    totals, count = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for _,x,y in window_batches(array,w,batch_size):
            x,y = torch.tensor(x,device=device),torch.tensor(y,device=device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            _,_,losses = model.optimize_pair(x,y)
            if optimizer is not None:
                losses.L_HDEG.backward()
                for name,parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise ValueError(f'Nonfinite gradient: {name}')
                optimizer.step()
            for name,value in losses.as_dict().items():
                totals[name] = totals.get(name,0.)+float(value.detach().cpu())*len(x)
            count += len(x)
    if count == 0:
        raise ValueError('No window pairs available.')
    return {'samples':count, **{name:value/count for name,value in totals.items()}}


def train(config, args):
    root, manifest, model, device = context(config)
    # Fixed split policy: never optimize on validation or test observations.
    if any(manifest['splits'][s]['pairs']==0 for s in ('train','val')):
        raise ValueError('End-to-end training requires both train and val pairs. Excerpts are insufficient.')
    if Path(args.run).name != args.run or args.run in ('','.','..'):
        raise ValueError('--run must be a single directory name.')
    destination = root/'training'/args.run
    if destination.exists():
        raise FileExistsError(f'{destination} exists; use a new training --run name.')
    options = config['training']
    epochs,batch_size,lr = int(options['epochs']),int(options['batch_size']),float(options['lr'])
    if epochs<1 or batch_size<1 or not np.isfinite(lr) or lr<=0:
        raise ValueError('epochs, batch_size and learning rate must be positive.')
    initialization = resolve(args.initialization) if args.initialization else root/'checkpoints/initialization.pt'
    if initialization.exists():
        payload = restore(initialization,model,config,root)
        if payload['status'] != 'untrained_initialization':
            raise ValueError('Restart training requires an initialization checkpoint, not an existing trained run.')
    else:
        raise FileNotFoundError('Run run_dbrl.py --initialize first to freeze the shared initialization.')
    train_array = split_array(root,manifest,'train')
    val_array = split_array(root,manifest,'val')
    destination.mkdir(parents=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    best, history = float('inf'), []
    write_json(destination/'run.json', {'status':'running','config':config,
        'initialization_sha256':digest(initialization),'dataset_sha256':digest(root/'dataset.json'),
        'selection':'minimum full validation L_HDEG; earliest epoch wins ties',
        'test_accessed':False,'command':sys.argv})
    for epoch in range(1,epochs+1):
        training = epoch_pass(model,train_array,manifest['window_size'],batch_size,device,optimizer)
        validation = epoch_pass(model,val_array,manifest['window_size'],batch_size,device)
        record = {'epoch':epoch,'train':training,'val':validation}
        history.append(record)
        write_json(destination/'history.json',history)
        payload = checkpoint_payload(model,config,root,'trained_last',epoch=epoch,
                    optimizer_state_dict=optimizer.state_dict(),metrics=record)
        atomic_checkpoint(destination/'last.pt',payload)
        if validation['L_HDEG'] < best:
            best = validation['L_HDEG']
            payload['status'] = 'trained_selected'
            atomic_checkpoint(destination/'best.pt',payload)
        print(json.dumps(record),flush=True)
    write_json(destination/'completion.json',{'status':'training_complete_not_scientifically_validated',
        'epochs':epochs,'best_validation_loss':best,'best_checkpoint_sha256':digest(destination/'best.pt'),
        'test_accessed':False})
    print(f'Selected checkpoint: {destination / "best.pt"}')
