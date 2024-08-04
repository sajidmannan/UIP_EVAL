# UIP_EVAL

## Training folder structure

`configs` contains the YAML configuration files used by the `matsciml` experiment parser.
At a high level, a single experiment YAML defines the scope of the experiment (e.g. task,
model, and dataset) imperatively. The experiment is then composed by passing definitions
for each component, i.e. a path to the LiPS dataset YAML file. An example call looks
like this:

```console
python matsciml/experiments/training_script.py \
	-e configs/experiments/faenet_lips_force.yaml \
	-m configs/models/faenet_pyg.yaml \
	-d configs/datasets/lips.yaml \
	-t configs/trainer.yaml
```

### Adding an experiment

1. Copy one of the experiment YAML configs; no hard and fast rule for naming scheme,
but to start off we have `<model>_<dataset>_<task>.yml` just for the ease of access.
2. Modify the keys in the experiment YAML config - the keys must match what are
defined in the other configs (e.g. `lips` refers to the name of the YAML file)
3. Update `trainer.yaml` as needed: _in particular, set the `wandb` entity to yours_!
4. Update the dataset YAML file as needed: pay attention to batch size, and paths.

### Common tweaking parameters

- Batch size (per DDP worker) is modified in the dataset YAML.
- Number of workers, epochs, callbacks are configured in `trainer.yaml`
- Learning rate is configured in the model YAML.
