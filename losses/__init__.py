from .topo_losses import (                  # noqa
    dice_loss,
    weighted_bce_loss,
    skeleton_recall_loss,
    cldice_loss,
    distance_transform_loss,
    edge_bce_loss,
    multi_scale_consistency_loss,
    TopoSAMLoss,
)
from .rwft_topo_losses import (              # noqa
    reliability_weighted_focal_tversky_loss,
    RWFTTopoSAMLoss,
)


def build_loss(cfg_loss: dict):
    """Factory: pick the loss class based on cfg['loss']['type'].

    type == "rwft"   -> RWFTTopoSAMLoss  (4-loss, redesigned)
    otherwise        -> TopoSAMLoss      (legacy 7-loss / lite)
    """
    if str(cfg_loss.get("type", "default")).lower() == "rwft":
        return RWFTTopoSAMLoss(cfg_loss)
    return TopoSAMLoss(cfg_loss)
