from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import Tensor, nn
from src.models.hdeg.dbrl import DBRL
from src.models.hdeg.bse import BehavioralStateEstimator
from src.models.hdeg.bil import BehavioralInteractionLearner
from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner
from src.models.hdeg.hbf import HierarchicalBehavioralForecaster
from src.models.hdeg.mo import ModelOptimization

class HDEGEndToEndModel(nn.Module):
    """Live trainable DBRL -> BSE -> BIL -> EBRL -> HBF -> MO graph."""

    def __init__(
        self,
        *,
        dbrl: DBRL,
        bse: BehavioralStateEstimator,
        bil: BehavioralInteractionLearner,
        ebrl: EcosystemBehavioralRepresentationLearner,
        hbf: HierarchicalBehavioralForecaster,
        mo: ModelOptimization,
        compatibility_mask: Tensor,
    ) -> None:
        super().__init__()
        self.dbrl = dbrl
        self.bse = bse
        self.bil = bil
        self.ebrl = ebrl
        self.hbf = hbf
        self.mo = mo
        self.register_buffer("compatibility_mask", compatibility_mask, persistent=True)

    def encode_window(self, x: Tensor) -> Dict[str, Tensor]:
        """Run one raw observation mini-batch through DBRL/BSE/BIL/EBRL."""
        Z = self.dbrl(x)
        S = self.bse(Z, self.compatibility_mask)
        S_tilde = self.bil(S)
        g = self.ebrl(S_tilde)
        return {"Z": Z, "S": S, "S_tilde": S_tilde, "g": g}

    def forecast(self, hierarchy_t: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.hbf(
            hierarchy_t["Z"],
            hierarchy_t["S"],
            hierarchy_t["S_tilde"],
            hierarchy_t["g"],
        )

    def optimize_pair(
        self,
        x_t: Tensor,
        x_t1: Tensor,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], object]:
        """
        Execute both sides of one paired window mini-batch.

        No branch is wrapped in no_grad and no representation is detached.
        """
        observed_t = self.encode_window(x_t)
        predicted_t1 = self.forecast(observed_t)
        observed_t1 = self.encode_window(x_t1)
        objectives = self.mo(observed_t1, predicted_t1)
        return observed_t, observed_t1, objectives


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_live_model(
    *,
    config: dict,
    behavioral_config,
    compatibility_mask: Tensor,
    device: torch.device,
) -> HDEGEndToEndModel:
    num_devices = int(behavioral_config.num_devices)
    num_states = int(behavioral_config.num_states)

    dbrl_cfg = config["hdeg"]["dbrl"]
    bse_cfg = config["hdeg"]["bse"]
    hbf_cfg = config["hdeg"]["hbf"]

    embedding_dim = int(dbrl_cfg["embedding_dim"])

    if int(dbrl_cfg["graph_top_k"]) > (
        num_devices if bool(dbrl_cfg["graph_self_loops"]) else num_devices - 1
    ):
        raise ValueError("DBRL graph_top_k exceeds the allowed device-neighbor count.")

    dbrl = DBRL(
        hidden_dim=int(dbrl_cfg["hidden_dim"]),
        embedding_dim=embedding_dim,
        gru_layers=int(dbrl_cfg["gru_layers"]),
        dropout=0.0,
        graph_top_k=int(dbrl_cfg["graph_top_k"]),
        graph_self_loops=bool(dbrl_cfg["graph_self_loops"]),
        graph_symmetric=bool(dbrl_cfg["graph_symmetric"]),
        graph_heads=int(dbrl_cfg["graph_heads"]),
        graph_dropout=0.0,
        # Constructor argument is not a DBRL architectural parameter; it is
        # retained by the frozen driver for explicit validation.
    )

    bse = BehavioralStateEstimator(
        num_states=num_states,
        embedding_dim=embedding_dim,
        num_heads=int(bse_cfg["num_heads"]),
    )

    # BIL's frozen CU topology is internal to the current implementation.
    if num_states != 9:
        raise ValueError(f"Frozen CU BIL expects K=9; received K={num_states}.")
    bil = BehavioralInteractionLearner(
        num_states=num_states,
        embedding_dim=embedding_dim,
    )

    ebrl = EcosystemBehavioralRepresentationLearner(
        num_states=num_states,
        embedding_dim=embedding_dim,
    )

    hbf = HierarchicalBehavioralForecaster(
        num_devices=num_devices,
        num_states=num_states,
        embedding_dim=embedding_dim,
        dynamics_hidden_dim=int(hbf_cfg["dynamics_hidden_dim"]),
    )

    mo_cfg = config.get("hdeg", {}).get("mo", {})
    mo = ModelOptimization(
        lambda_z=float(mo_cfg.get("lambda_z", 1.0)),
        lambda_s=float(mo_cfg.get("lambda_s", 1.0)),
        lambda_s_tilde=float(mo_cfg.get("lambda_s_tilde", 1.0)),
        lambda_g=float(mo_cfg.get("lambda_g", 1.0)),
    )

    model = HDEGEndToEndModel(
        dbrl=dbrl,
        bse=bse,
        bil=bil,
        ebrl=ebrl,
        hbf=hbf,
        mo=mo,
        compatibility_mask=compatibility_mask,
    ).to(device)

    return model


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


