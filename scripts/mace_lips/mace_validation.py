from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ForceRegressionTask

"""
This script acts as an intermediate step for validating the trained
MACE model on LiPS.

We download the uploaded checkpoint with the lowest validation force
error, load the checkpoint into a `ForceRegressionTask`, then
run through the validation set with the same loading pipeline. The
saved checkpoint weights correspond to the exponential moving averaged
ones.

After going through the full validation set, we push the predicted
and ground truth values to the initialized `wandb` run.
"""

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
    """Simple utility function to move things to correct device"""
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
    # run forward pass
    outputs = task(batch)
    energies = outputs["energy"].detach().cpu().numpy()
    forces = outputs["force"].detach().cpu().numpy()
    pred_energies.append(energies)
    pred_forces.append(forces)
    # save the ground truth labels as well
    true_energies.append(batch["targets"]["energy"].cpu().numpy())
    true_forces.append(batch["targets"]["force"].cpu().numpy())

# create a wandb artifact object to stash the results to
infer_art = wandb.Artifact(name="mace-uip-validation", type="result")

for array, name in zip(
    [pred_energies, pred_forces, true_energies, true_forces],
    ["pred_energies", "pred_forces", "true_energies", "true_forces"],
):
    output_path = artifact_dir.joinpath(name).with_suffix(".npy")
    array = np.vstack(array)
    np.save(output_path, array)
    # somewhat annoyingly, this omits the file extension when pushed
    infer_art.add_file(local_path=output_path, name=name)

run.log_artifact(infer_art)
