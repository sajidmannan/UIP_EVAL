from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
import numpy as np

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ForceRegressionTask

pl.seed_everything(215125)
torch.set_float32_matmul_precision("medium")

run = wandb.init(
    project="matsciml-uip-eval", tags=["inference", "validation", "results"]
)

artifact = run.use_artifact(
    "laserkelvin/matsciml-uip-eval/model-pfs05aqp:v74", type="model"
)
artifact_dir = Path(artifact.download())

task = ForceRegressionTask.load_from_checkpoint(artifact_dir.joinpath("model.ckpt"))

# move task to device
task = task.to("cuda")

ROOT_DIR = "/datasets-alt/molecular-data/lips"

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

# manual inference
dm.setup("fit")
val_loader = dm.val_dataloader()


def to(data, device):
    new_dict = {}
    for key, value in data.items():
        if hasattr(value, "to"):
            new_dict[key] = value.to(device)
        else:
            new_dict[key] = value
    return new_dict


pred_energies = []
true_energies = []
pred_forces = []
true_forces = []

for index, batch in enumerate(val_loader):
    # make sure we don't contaminate
    task.zero_grad(True)
    batch = to(batch, task.device)
    outputs = task(batch)
    energies = outputs["energy"].detach().cpu().numpy()
    forces = outputs["force"].detach().cpu().numpy()
    pred_energies.append(energies)
    pred_forces.append(forces)
    true_energies.append(batch["targets"]["energy"].cpu().numpy())
    true_forces.append(batch["targets"]["force"].cpu().numpy())

infer_art = wandb.Artifact(name="mace-uip-validation", type="result")

for array, name in zip(
    [pred_energies, pred_forces, true_energies, true_forces],
    ["pred_energies", "pred_forces", "true_energies", "true_forces"],
):
    output_path = artifact_dir.joinpath(name).with_suffix(".npy")
    array = np.vstack(array)
    np.save(output_path, array)
    infer_art.add_file(local_path=output_path, name=name)

run.log_artifact(infer_art)
