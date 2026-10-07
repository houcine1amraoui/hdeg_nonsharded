import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from src.pipeline.data import config_from, prepare, semantic_devices, resolve, load_dataset, split_array, window_batches


def fixture(directory):
    cfg = config_from('configs/restart.yaml')
    cfg['data']['csv'] = str(directory/'input.csv')
    cfg['data']['output'] = str(directory/'prepared')
    cfg['data']['window_size'] = 3
    cfg['training'].update(epochs=2,batch_size=16,device='cpu')
    cfg['hdeg']['dbrl'].update(hidden_dim=4,embedding_dim=4,graph_top_k=2)
    cfg['hdeg']['hbf']['dynamics_hidden_dim'] = 8
    devices = semantic_devices(resolve(cfg['data']['semantic_config']))
    frames = []
    rng = np.random.default_rng(42)
    for name, (start,end) in cfg['data']['periods'].items():
        values = rng.normal(size=(40,len(devices)))
        values += 0 if name=='actor1_t1' else 100
        part = pd.DataFrame(values, columns=devices)
        part.insert(0,'Timestamp',pd.date_range(start,periods=40,freq='s'))
        frames.append(part)
    pd.concat(frames,ignore_index=True).to_csv(cfg['data']['csv'],index=False)
    return cfg


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = fixture(Path(self.temp.name))

    def test_scaler_uses_train_only_and_exact_device_schema(self):
        manifest = prepare(self.cfg)
        root, _ = load_dataset(self.cfg)
        self.assertEqual(len(manifest['devices']),23)
        train = split_array(root,manifest,'train')
        test = split_array(root,manifest,'actor2_test')
        self.assertTrue(np.allclose(train.min(0),0))
        self.assertTrue(np.allclose(train.max(0),1))
        self.assertGreater(float(test.min()),1)
        self.assertEqual(manifest['splits']['train']['rows'],36)
        self.assertEqual(manifest['splits']['val']['rows'],4)

    def test_lazy_windows_include_last_target(self):
        raw = np.arange(36,dtype=np.float32).reshape(12,3)
        batches = list(window_batches(raw,3,4))
        x = np.concatenate([b[1] for b in batches])
        y = np.concatenate([b[2] for b in batches])
        self.assertEqual(len(x),9)
        np.testing.assert_array_equal(x[-1],raw[8:11])
        np.testing.assert_array_equal(y[-1],raw[9:12])
        np.testing.assert_array_equal(x[:,1:],y[:,:-1])
        self.assertEqual(sum(len(b[1]) for b in window_batches(raw,3,4,paired=False)),10)

    def test_missing_periods_rejected_unless_explicit_module_check(self):
        df = pd.read_csv(self.cfg['data']['csv']).iloc[:40]
        df.to_csv(self.cfg['data']['csv'],index=False)
        with self.assertRaisesRegex(ValueError,'No paired windows'):
            prepare(self.cfg)
        result = prepare(self.cfg,allow_incomplete=True)
        self.assertTrue(result['incomplete'])
        self.assertEqual(result['splits']['actor2_test']['rows'],0)

    def test_no_windows_across_time_gaps(self):
        df = pd.read_csv(self.cfg['data']['csv']).drop(index=5)
        df.to_csv(self.cfg['data']['csv'],index=False)
        with self.assertRaisesRegex(ValueError,'time gaps'):
            prepare(self.cfg)

    def test_overlap_rejected(self):
        self.cfg['data']['periods']['actor2_test'] = self.cfg['data']['periods']['actor1_t1']
        with self.assertRaisesRegex(ValueError,'non-overlapping'):
            prepare(self.cfg)

    def test_data_and_timestamp_mutation_detected(self):
        manifest = prepare(self.cfg)
        root,_ = load_dataset(self.cfg)
        p = root/'train_timestamps.npy'
        ticks = np.load(p)
        ticks[0] += np.timedelta64(1,'s')
        np.save(p,ticks)
        with self.assertRaisesRegex(ValueError,'timestamps differ'):
            split_array(root,manifest,'train')

    def test_config_change_rejected(self):
        prepare(self.cfg)
        self.cfg['data']['val_ratio'] = 0.2
        with self.assertRaisesRegex(ValueError,'configuration changed'):
            load_dataset(self.cfg)

    def test_existing_output_preserved(self):
        prepare(self.cfg)
        with self.assertRaises(FileExistsError):
            prepare(self.cfg)


HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torch_geometric') is not None


@unittest.skipUnless(HAS_TORCH,'Requires PyTorch and torch-geometric; not a passed test when skipped.')
class NeuralIntegrationChecks(unittest.TestCase):
    def test_stage_exports_training_and_checkpoint_consistency(self):
        import torch
        from src.pipeline.runtime import export_stage,train,context,restore
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            cfg = fixture(Path(directory))
            prepare(cfg)
            args = SimpleNamespace(checkpoint=None,initialize=True,require_trained=False,run='checks',split='train')
            export_stage(cfg,args,'dbrl')
            args.initialize = False
            for stage in ['bse','bil','ebrl','hbf','mo','mbai']:
                export_stage(cfg,args,stage)
            root,manifest,model,device = context(cfg)
            checkpoint = root/'checkpoints/initialization.pt'
            restore(checkpoint,model,cfg,root)
            model.eval()
            raw = split_array(root,manifest,'train')
            _,x,_ = next(window_batches(raw,3,4,paired=False))
            with torch.no_grad():
                expected = model.encode_window(torch.tensor(x,device=device))
            for stage,key in [('dbrl','Z'),('bse','S'),('bil','S_tilde'),('ebrl','g')]:
                actual = np.load(root/f'exports/checks/train/{stage}/{key}.npy')[:4]
                np.testing.assert_allclose(actual,expected[key].cpu().numpy(),rtol=1e-5,atol=1e-6)
            # Forecast and aligned scoring preserve every paired target.
            scores=np.load(root/'exports/checks/train/mbai/A.npy')
            self.assertEqual(len(scores),manifest['splits']['train']['pairs'])
            args.require_trained=True
            args.run='invalid_initialization'
            with self.assertRaisesRegex(ValueError,'selected trained'):
                export_stage(cfg,args,'dbrl')
            train(cfg,SimpleNamespace(run='test_run',initialization=None))
            history=json.loads((root/'training/test_run/history.json').read_text())
            payload=torch.load(root/'training/test_run/best.pt',weights_only=False)
            self.assertEqual(payload['epoch'],min(history,key=lambda h:h['val']['L_HDEG'])['epoch'])
            self.assertEqual(payload['status'],'trained_selected')
            args.checkpoint=str(root/'training/test_run/best.pt')
            args.run='trained'
            export_stage(cfg,args,'dbrl')
            # An existing upstream artifact from a different checkpoint is refused.
            args.run='checks'
            from shutil import copytree
            copytree(root/'exports/checks/train/dbrl',root/'exports/mixed/train/dbrl')
            args.run='mixed'
            with self.assertRaisesRegex(ValueError,'checkpoint/dataset/split mismatch'):
                export_stage(cfg,args,'bse')


if __name__ == '__main__':
    unittest.main()
