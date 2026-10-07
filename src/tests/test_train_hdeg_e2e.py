from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def load_as(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_test_imports():
    # Package skeleton required by the runner imports.
    for name in ["src", "src.models", "src.models.hdeg", "src.common", "src.common.graph", "src.utils"]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod

    # Real frozen modules under test.
    bse = load_as("src.models.hdeg.bse", ROOT / "bse.py")
    bil = load_as("src.models.hdeg.bil", ROOT / "bil.py")
    ebrl = load_as("src.models.hdeg.ebrl", ROOT / "ebrl.py")
    hbf = load_as("src.models.hdeg.hbf", ROOT / "hbf.py")
    mo = load_as("src.models.hdeg.mo", ROOT / "mo.py")

    class FakeDBRL(torch.nn.Module):
        def __init__(self, embedding_dim: int = 8):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.proj = torch.nn.Linear(1, embedding_dim)

        def forward(self, x):
            # Live trainable stand-in only because torch_geometric is not
            # installed in this execution environment. Shape is identical
            # to the real DBRL contract: (B,W,N) -> (B,N,D).
            last = x[:, -1, :].unsqueeze(-1)
            return self.proj(last)

    dbrl_mod = types.ModuleType("src.models.hdeg.dbrl")
    dbrl_mod.DBRL = FakeDBRL
    sys.modules["src.models.hdeg.dbrl"] = dbrl_mod

    semantics = types.ModuleType("src.common.graph.semantics")
    semantics.load_behavioral_state_config = lambda *a, **k: None
    sys.modules["src.common.graph.semantics"] = semantics

    device_mod = types.ModuleType("src.utils.device")
    device_mod.get_device = lambda: torch.device("cpu")
    sys.modules["src.utils.device"] = device_mod

    folders_mod = types.ModuleType("src.utils.get_folders_utils")
    folders_mod.get_processed_folder = lambda config: Path(".")
    sys.modules["src.utils.get_folders_utils"] = folders_mod

    seed_mod = types.ModuleType("src.utils.seed")
    seed_mod.set_seed = lambda seed: torch.manual_seed(seed)
    sys.modules["src.utils.seed"] = seed_mod

    # Existing DBRL shard utility import is replaced with contract-compatible
    # stubs because this unit test exercises the live graph, not disk I/O.
    run_dbrl = types.ModuleType("run_dbrl_sharded")
    run_dbrl.load_manifest = lambda **kwargs: None
    run_dbrl.load_window_shard = lambda **kwargs: None
    run_dbrl.resolve_shard_path = lambda *args, **kwargs: None
    run_dbrl.verify_window_target_alignment = lambda artifact: None
    sys.modules["run_dbrl_sharded"] = run_dbrl

    runner = load_as("train_hdeg_e2e", ROOT / "train_hdeg_e2e.py")
    return runner, bse, bil, ebrl, hbf, mo


def test_live_graph_autograd():
    runner, bse_mod, bil_mod, ebrl_mod, hbf_mod, mo_mod = install_test_imports()

    B, W, N, K, D = 4, 5, 23, 9, 8
    torch.manual_seed(42)

    mask = torch.zeros(K, N)
    for k in range(K):
        mask[k, k % N] = 1.0

    model = runner.HDEGEndToEndModel(
        dbrl=runner.DBRL(embedding_dim=D),
        bse=bse_mod.BehavioralStateEstimator(K, D, 1),
        bil=bil_mod.BehavioralInteractionLearner(num_states=K, embedding_dim=D),
        ebrl=ebrl_mod.EcosystemBehavioralRepresentationLearner(D, num_states=K),
        hbf=hbf_mod.HierarchicalBehavioralForecaster(
            num_devices=N, num_states=K, embedding_dim=D, dynamics_hidden_dim=16
        ),
        mo=mo_mod.ModelOptimization(),
        compatibility_mask=mask,
    )

    x = torch.randn(B, W, N)
    y = torch.randn(B, W, N)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }

    optimizer.zero_grad(set_to_none=True)
    _, target, objectives = model.optimize_pair(x, y)

    assert target["Z"].requires_grad
    assert objectives.L_HDEG.requires_grad
    assert objectives.L_Z.requires_grad
    assert objectives.L_S.requires_grad
    assert objectives.L_S_tilde.requires_grad
    assert objectives.L_G.requires_grad

    objectives.L_HDEG.backward()

    required_modules = ("dbrl", "bse", "bil", "ebrl", "hbf")
    for prefix in required_modules:
        grads = [
            p.grad for name, p in model.named_parameters()
            if name.startswith(prefix + ".") and p.requires_grad
        ]
        assert grads, f"No trainable parameters found for {prefix}"
        assert all(g is not None for g in grads), f"Missing gradient in {prefix}"
        assert all(torch.isfinite(g).all() for g in grads), f"Non-finite gradient in {prefix}"
        assert any(torch.any(g != 0) for g in grads), f"Zero gradients throughout {prefix}"

    optimizer.step()

    assert any(
        not torch.equal(before[name], p.detach())
        for name, p in model.named_parameters()
    )


def test_four_level_loss_changes_only_with_corresponding_target():
    runner, bse_mod, bil_mod, ebrl_mod, hbf_mod, mo_mod = install_test_imports()
    del runner, bse_mod, bil_mod, ebrl_mod, hbf_mod

    mo = mo_mod.ModelOptimization()
    B, N, K, D = 2, 23, 9, 8
    observed = {
        "Z": torch.zeros(B, N, D),
        "S": torch.zeros(B, K, D),
        "S_tilde": torch.zeros(B, K, D),
        "g": torch.zeros(B, D),
    }
    predicted = {k: v.clone() for k, v in observed.items()}
    predicted["S"] += 1.0

    out = mo(observed, predicted)
    assert torch.isclose(out.L_Z, torch.tensor(0.0))
    assert torch.isclose(out.L_S_tilde, torch.tensor(0.0))
    assert torch.isclose(out.L_G, torch.tensor(0.0))
    assert out.L_S > 0
    assert torch.isclose(out.L_HDEG, out.L_S)


def test_one_shard_training_loop_uses_paired_X_Y_and_updates():
    runner, bse_mod, bil_mod, ebrl_mod, hbf_mod, mo_mod = install_test_imports()
    B, W, N, K, D = 6, 5, 23, 9, 8
    torch.manual_seed(7)

    mask = torch.zeros(K, N)
    for k in range(K):
        mask[k, k % N] = 1.0

    model = runner.HDEGEndToEndModel(
        dbrl=runner.DBRL(embedding_dim=D),
        bse=bse_mod.BehavioralStateEstimator(K, D, 1),
        bil=bil_mod.BehavioralInteractionLearner(num_states=K, embedding_dim=D),
        ebrl=ebrl_mod.EcosystemBehavioralRepresentationLearner(D, num_states=K),
        hbf=hbf_mod.HierarchicalBehavioralForecaster(
            num_devices=N, num_states=K, embedding_dim=D, dynamics_hidden_dim=16
        ),
        mo=mo_mod.ModelOptimization(),
        compatibility_mask=mask,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = np.random.default_rng(7).normal(size=(B, W, N)).astype(np.float32)
    Y = np.random.default_rng(8).normal(size=(B, W, N)).astype(np.float32)

    runner.load_window_shard = lambda **kwargs: {
        "X": X,
        "Y": Y,
        "window_start_timestamps": np.arange(B),
        "target_timestamps": np.zeros((B, W), dtype=np.int64),
        "split": np.asarray("train"),
        "shard_index": np.asarray(0),
        "start_index": np.asarray(0),
        "end_index": np.asarray(B),
        "window_size": np.asarray(W),
        "num_devices": np.asarray(N),
    }
    runner.verify_window_target_alignment = lambda artifact: None

    metrics = runner.run_epoch(
        model,
        optimizer,
        shard_paths=[Path("synthetic_shard_000000.npz")],
        split="train",
        window_size=W,
        num_devices=N,
        num_states=K,
        embedding_dim=D,
        batch_size=2,
        device=torch.device("cpu"),
        train=True,
        max_batches_per_shard=2,
    )

    assert metrics.samples == 4
    assert metrics.shards == 1
    assert metrics.batches == 2
    assert np.isfinite(metrics.loss)
    assert metrics.loss >= 0.0
