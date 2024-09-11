import pytorch_lightning as pl
import torch
from e3nn.o3 import Irreps
from mace.modules import ScaleShiftMACE
from mace.modules.blocks import RealAgnosticResidualInteractionBlock
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from matsciml.models.base import ForceRegressionTask
from matsciml.lightning.callbacks import (
    ExponentialMovingAverageCallback,
    ManualGradientClip,
)
from matsciml.models.pyg.mace import MACEWrapper

"""
This script is used to reproduce the MACE training run on the
full LiPS dataset using the matsciml pipeline.

The run should be reproducible using public matsciml#940575c,
and should be entirely self-contained.

Notable things:
    1. Uses a variety of callbacks, particularly gradient clipping
    and exponential moving average weights.
    2. Periodic boundary conditions with a cut off of 5
    3. Logs to weights and biases
    4. Sets medium precision for single precision; uses tensor cores at
    lower precision (i.e. FP32 = combined FP16) but improves throughput
"""

pl.seed_everything(215125)
# use tensor cores
torch.set_float32_matmul_precision("medium")

available_models = {
    "mace": {
        "encoder_class": MACEWrapper,
        "encoder_kwargs": {
            "mace_module": ScaleShiftMACE,
            "num_atom_embedding": 100,  # this is set to 100 and will use ion energies
            "r_max": 5.0,
            "num_bessel": 8,
            "num_polynomial_cutoff": 5.0,
            "max_ell": 3,
            "interaction_cls": RealAgnosticResidualInteractionBlock,
            "interaction_cls_first": RealAgnosticResidualInteractionBlock,
            "num_interactions": 2,
            "hidden_irreps": Irreps("128x0e + 128x1o"),
            "atom_embedding_dim": 16,
            "MLP_irreps": Irreps("16x0e"),
            "avg_num_neighbors": 25.188983917236328,
            "correlation": 3,
            "radial_type": "bessel",
            "gate": nn.SiLU(),
            "atomic_inter_scale": 0.610558,
            "atomic_inter_shift": 0,
            "distance_transform": None,
        },
        # note we are using the output heads for the task - not outputs from MACE!
        "output_kwargs": {"lazy": False, "input_dim": 640, "hidden_dim": 640},
        "task_loss_scaling": {"energy": 1, "force": 10},
    }
}

ROOT_DIR = "/datasets-alt/molecular-data/lips"
task = ForceRegressionTask(**available_models["mace"])

dm = MatSciMLDataModule(
    "LiPSDataset",
    train_path=f"{ROOT_DIR}/train",
    val_split=f"{ROOT_DIR}/val",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(5.0, adaptive_cutoff=True),
            PointCloudToGraphTransform(
                "pyg",
                node_keys=["pos", "atomic_numbers"],
            ),
        ],
    },
    batch_size=16,
    num_workers=8,
)

save_dir = "./wandb_logs"
wb_logger = WandbLogger(
    log_model="all",
    save_dir=save_dir,
    project="matsciml-uip-eval",
    entity="laserkelvin",
    mode="online",
)

trainer = pl.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=100,
    logger=wb_logger,
    callbacks=[
        StochasticWeightAveraging(
            swa_lrs=1e-2, swa_epoch_start=0.6, annealing_epochs=30
        ),
        ExponentialMovingAverageCallback(decay=0.99),
        ManualGradientClip(10.0),
        LearningRateMonitor("step"),
        ModelCheckpoint(monitor="val_force_epoch", save_top_k=5),
    ],
)
trainer.fit(task, datamodule=dm)
