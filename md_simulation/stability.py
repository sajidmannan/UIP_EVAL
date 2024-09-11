import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import units
from ase.md import MDLogger
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list
from checkpoint import multitask_from_checkpoint
from loguru import logger
from matsciml.datasets.transforms import (
    FrameAveraging,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning import MatSciMLDataModule
from tqdm import tqdm
from Utils import (
    ASEcalculator,
    get_initial_rdf,
    get_bond_lengths_noise,
    symmetricize_replicate,
    replicate_system,
    get_bond_lengths_TrajAvg,
    convBatchtoAtoms,
    minimize_structure,
    get_rdf,
)


class StabilityException(Exception):
    pass


def run_simulation(atoms, runsteps=1000, SimDir="./"):
    traj = []
    logger.info("Calculating initial RDFs ... ")
    _, initial_rdf = get_initial_rdf(
        atoms, perturb=20, noise_std=0.05, max_atoms=config.max_atoms, replicate=True
    )
    initial_bond_lengths, Initial_Pair_rdfs = get_bond_lengths_noise(
        atoms,
        perturb=20,
        noise_std=0.05,
        max_atoms=config.max_atoms,
        r_max=config.rdf_r_max,
    )
    initial_temperature = config.temp
    # Replicate_system
    replication_factors, size = symmetricize_replicate(
        len(atoms),
        max_atoms=config.max_atoms,
        box_lengths=atoms.get_cell_lengths_and_angles()[:3],
    )
    atoms = replicate_system(atoms, replication_factors)

    # Set_calculator

    Loaded_model = multitask_from_checkpoint(config.model_path)
    calculator = ASEcalculator(Loaded_model, config.model_name)

    # calculator = MACECalculator(model_path=config.model_path, device=config.device, default_dtype='float64')
    atoms.set_calculator(calculator)
    atoms = minimize_structure(atoms, steps=config.minimize_steps)

    # Set_simulation
    # NVE
    # MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature)
    # initial_energy = atoms.get_total_energy()
    # dyn = VelocityVerlet(atoms, dt=timestep * units.fs)

    # NPT
    MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature)
    dyn = NPTBerendsen(
        atoms,
        timestep=config.timestep * units.fs,
        temperature_K=config.temperature,
        pressure_au=config.pressure * units.bar,
        compressibility_au=4.57e-5 / units.bar,
    )

    dyn.attach(
        MDLogger(
            dyn,
            atoms,
            os.path.join(SimDir, "Simulation_thermo.log"),
            header=True,
            stress=True,
            peratom=False,
            mode="w",
        ),
        interval=config.thermo_interval,
    )

    def write_frame(a=atoms):
        if SimDir is not None:
            a.write(
                os.path.join(SimDir, f"MD_{atoms.get_chemical_formula()}_NPT.xyz"),
                append=True,
            )

    dyn.attach(write_frame, interval=config.trajdump_interval)

    def append_traj(a=atoms):
        traj.append(a.copy())

    dyn.attach(append_traj, interval=1)

    # def energy_stability(a=atoms):
    #     logger.info("Checking energy stability...", end='\t')
    #     current_energy = atoms.get_total_energy()
    #     energy_error = abs((current_energy - initial_energy) / initial_energy)
    #     if energy_error > config.energy_tolerence:
    #         logger.error(f"Unstable : Energy_error={energy_error:.6g} (> {config.energy_tolerence:.6g})")
    #         raise StabilityException("Energy_criterion violated. Stopping the simulation.")
    #     else:
    #         logger.info(f"Stable : Energy_error={energy_error:.6g} (< {config.energy_tolerence:.6g})")

    # dyn.attach(energy_stability, interval=config.energy_criteria_interval)

    def temperature_stability(atoms, initial_temperature, temperature_tolerance):
        if len(traj) >= config.initial_equilibration_period:
            logger.info("Checking temperature stability...", end="\t")
            current_temperature = atoms.get_temperature()
            temperature_error = abs(
                (current_temperature - initial_temperature) / initial_temperature
            )
            if temperature_error > temperature_tolerance:
                logger.error(
                    f"Unstable : Temperature_error={temperature_error:.6g} (> {temperature_tolerance:.6g})"
                )
                raise StabilityException(
                    "Temperature criterion violated. Stopping the simulation."
                )
            else:
                logger.info(
                    f"Stable : Temperature_error={temperature_error:.6g} (< {temperature_tolerance:.6g})"
                )

    # Attach the temperature stability check to the dynamics object
    dyn.attach(
        temperature_stability,
        interval=config.temperature_criteria_interval,
        atoms=atoms,
        initial_temperature=initial_temperature,
        temperature_tolerance=config.temperature_tolerance,
    )

    def calculate_rmsd(traj):
        initial_positions = traj[0].get_positions()
        N = len(traj[0])
        T = len(traj)
        displacements = np.zeros((N, T, 3))
        for t in range(T):
            current_positions = traj[t].get_positions()
            displacements[:, t, :] = current_positions - initial_positions
        msd = np.mean(np.sum(displacements**2, axis=2), axis=1)
        rmsd = np.sqrt(msd)
        return rmsd

    def calculate_average_nn_distance(atoms):
        i, j, _ = neighbor_list("ijd", atoms, cutoff=5.0)
        distances = atoms.get_distances(i, j, mic=True)
        return np.mean(distances)

    def lindemann_stability(a=atoms):
        if len(traj) >= config.lindemann_traj_length:
            logger.info("Checking lindemann stability...", end="\t")
            rmsd = calculate_rmsd(traj[-config.lindemann_traj_length :])
            avg_nn_distance = calculate_average_nn_distance(traj[0])
            lindemann_coefficient = np.mean(rmsd) / avg_nn_distance
            if lindemann_coefficient > config.max_linedmann_coefficient:
                logger.error(
                    f"Unstable : Lindemann_coefficient={lindemann_coefficient:.6g} (> {config.max_linedmann_coefficient:.6g})"
                )
                logger.error(
                    f"Lindemann_stability criterion violated {lindemann_coefficient:.6g} > {config.max_linedmann_coefficient:.6g}, Stopping the simulation."
                )
                raise StabilityException()
            else:
                logger.info(
                    f"Stable : Lindemann_coefficient={lindemann_coefficient:.6g} (< {config.max_linedmann_coefficient:.6g})"
                )

    dyn.attach(lindemann_stability, interval=config.lindemann_criteria_interval)

    def rdf_stability(a=atoms):
        if len(traj) >= config.rdf_traj_length:
            logger.info("Checking RDF stability...", end="\t")
            r, rdf = get_rdf(traj[-config.rdf_traj_length :], r_max=config.rdf_r_max)
            RDF_len = min(len(rdf), len(initial_rdf))
            r = r[:RDF_len]
            rdf = rdf[:RDF_len]
            initial_rdf_ = initial_rdf[:RDF_len]
            error_rdf = (
                100
                * (((rdf - initial_rdf_) ** 2).sum())
                / (((initial_rdf_) ** 2).sum())
            )

            # Plotting the RDF
            plt.figure()
            plt.plot(r, initial_rdf_, label="Initial RDF")
            plt.plot(r, rdf, label="Simulated RDF")
            plt.xlabel("Distance (r)")
            plt.ylabel("RDF")
            plt.legend()
            plt.title(f"RDF Comparison\nInitial vs Simulated\nError={error_rdf:.6g}")
            plot_path = os.path.join(
                SimDir, f"RDF_{atoms.get_chemical_formula()}_{len(traj)}.png"
            )
            plt.savefig(plot_path)
            logger.info("Saved figure at {}", plot_path)
            plt.close()
            if error_rdf > config.max_rdf_error_percent:
                logger.error(
                    f"Unstable : RDF Error={error_rdf:.6g} (> {config.max_rdf_error_percent:.6g})"
                )
                logger.error(
                    f"RDF criterion violated. Stopping the simulation. WF={error_rdf:.6g}"
                )
                raise StabilityException()
            else:
                logger.info(
                    f"Stable : RDF Error={error_rdf:.6g} (< {config.max_rdf_error_percent:.6g})"
                )

    dyn.attach(rdf_stability, interval=config.rdf_criteria_interval)

    def bond_lengths_stability(a=atoms):
        if len(traj) >= config.lindemann_traj_length:
            logger.info("Checking Bonds stability...", end="\t")
            curr_bond_lengths, Pair_rdfs = get_bond_lengths_TrajAvg(
                traj[-config.rdf_traj_length :], r_max=config.rdf_r_max
            )
            for key in curr_bond_lengths.keys():
                r, initial_rdf = Initial_Pair_rdfs[key]
                r, rdf = Pair_rdfs[key]
                RDF_len = min(len(rdf), len(initial_rdf))
                r = r[:RDF_len]
                rdf = rdf[:RDF_len]
                initial_rdf_ = initial_rdf[:RDF_len]
                error_percent = (
                    100
                    * (((rdf - initial_rdf_) ** 2).sum())
                    / (((initial_rdf_) ** 2).sum())
                )

                plt.figure()
                plt.plot(r, initial_rdf_, label="Initial RDF")
                plt.plot(r, rdf, label="Simulated RDF")
                plt.xlabel("Distance (r)")
                plt.ylabel("RDF")
                plt.legend()
                plt.title(
                    f"RDF Comparison: Bond {key}={curr_bond_lengths[key]:.6g}, Initial={initial_bond_lengths[key]:.6g}, Error={error_percent:.6g}"
                )
                plot_path = os.path.join(
                    SimDir,
                    f"PartialRDF_{atoms.get_chemical_formula()}_{key}_{len(traj)}.png",
                )
                plt.savefig(plot_path)
                logger.info("Saved figure at {}", plot_path)
                if False:  # error_percent > config.max_bond_error_percent:
                    logger.error(
                        f"Unstable : Bond {key}={curr_bond_lengths[key]:.6g}, Initial={initial_bond_lengths[key]:.6g}, Error={error_percent:.6g} (> {config.max_bond_error_percent:.6g})"
                    )
                    logger.error(
                        f"Bond length stability violated. Stopping the simulation. Bond {key}={curr_bond_lengths[key]:.6g}, Initial={initial_bond_lengths[key]:.6g}"
                    )
                    raise StabilityException()
                else:
                    logger.info(
                        f"Stable : Bond {key}: {error_percent: .6g} < {config.max_bond_error_percent:.6g} % Error"
                    )

    dyn.attach(bond_lengths_stability, interval=config.rdf_criteria_interval)

    try:
        logger.info(
            f"Simulating {atoms.get_chemical_formula()} {len(atoms)} atoms system ...."
        )
        counter = 0
        for k in tqdm(range(runsteps)):
            dyn.run(1)
            counter += 1
        return runsteps  # Simulation completed successfully
    except StabilityException:
        logger.error(
            f"Simulation of {atoms.get_chemical_formula()} {len(atoms)} atoms system failed after {counter} steps"
        )
        return len(traj)  # Return the number of steps completed before failure


class TestArgs:
    runsteps = 50000
    model_path = "/home/m3rg2000/Simulation/checkpoints-2024/FAENet_250k.ckpt"
    model_name = "faenet"  ##[tensornet, faenet, mace]
    data_path = "/home/m3rg2000/Universal_matscimal/Data/stability_new"
    timestep = 1.0
    temp = 298
    out_dir = "/home/m3rg2000/Universal_matscimal/Sim_output/"
    device = "cuda"
    replicate = True
    max_atoms = 200  # Replicate upto max_atoms (Min. will be max_atoms/2) (#Won't reduce if more than max_atoms)
    # energy_tolerence=0.1
    # energy_criteria_interval=100
    max_linedmann_coefficient = 0.3
    lindemann_criteria_interval = 1000
    lindemann_traj_length = 1000
    max_rdf_error_percent = 80
    max_bond_error_percent = 80
    bond_criteria_interval = 1000
    rdf_dr = 0.02
    rdf_r_max = 6.0
    rdf_traj_length = 1000
    rdf_criteria_interval = 1000
    trajdump_interval = 10
    minimize_steps = 200
    temperature = 300
    temperature_tolerance = 0.8
    thermo_interval = 10
    pressure = 1.01325
    temperature_criteria_interval = 1000
    initial_equilibration_period = 3000


# config=TestArgs()


def main(args, config):
    transforms = []

    if config.model_name == "faenet":
        transforms += [FrameAveraging(frame_averaging="3D", fa_method="stochastic")]

    # Load Data
    if config.model_name == "tensornet":
        graph_type = "dgl"
    else:
        graph_type = "pyg"
    dm = MatSciMLDataModule(
        "MaterialsProjectDataset",
        train_path=config.data_path,
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(cutoff_radius=6.0, adaptive_cutoff=True),
                PointCloudToGraphTransform(
                    graph_type,
                    node_keys=["pos", "atomic_numbers"],
                ),
            ]
            + transforms,
        },
        batch_size=1,
    )

    dm.setup()
    train_loader = dm.train_dataloader()
    # dataset_iter = iter(train_loader)

    time_steps = []
    # unreadable_files = []
    # Range = [0, 120]

    index = int(args.index)
    print("Index:", index)

    counter_batch = 0
    for batch in train_loader:
        if counter_batch == index:
            atoms = convBatchtoAtoms(batch)
            SimDir = os.path.join(
                config.out_dir, f"Simulation_{index}_{atoms.get_chemical_formula()}"
            )
            os.makedirs(SimDir, exist_ok=True)
            # Initialize logger
            logger.add(os.path.join(SimDir, "simulation.log"), rotation="500 MB")
            logger.info("All seeds set!")
            steps_completed = run_simulation(atoms, config.runsteps, SimDir)
            time_steps.append(steps_completed)
            logger.info(
                "System: {} : {} with originally {} atoms stopped at {} steps",
                counter_batch,
                atoms.get_chemical_formula(),
                len(atoms),
                steps_completed,
            )
            counter_batch += 1
        else:
            counter_batch += 1
            continue

    logger.info("Completed...")
    logger.info("Time Steps: {}", time_steps)


if __name__ == "__main__":
    config = TestArgs()
    # Seed for the Python random module
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)  # if you are using multi-GPU.
    parser = argparse.ArgumentParser(description="Run MD simulation with MACE model")
    parser.add_argument("--index", type=int, default=0, help="index of folder")
    # parser.add_argument("--init_conf_path", type=str, default="example/lips20/data/test/botnet.xyz", help="Path to the initial configuration")
    # parser.add_argument("--device", type=str, default="cuda", help="Device: ['cpu', 'cuda']")
    # parser.add_argument("--input_dir", type=str, default="./", help="folder path")
    # parser.add_argument("--out_dir", type=str, default="out_dir_sl/neqip/lips20/exp.csv", help="Output path")
    # parser.add_argument("--results_dir", type=str, default="out_dir_sl/neqip/lips20/", help="Output  directory path")

    # parser.add_argument("--temp", type=float, default=300, help="Temperature in Kelvin")
    # parser.add_argument("--pressure", type=float, default=1, help="pressure in atm")
    # parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs units")
    # parser.add_argument("--runsteps", type=int, default=1000, help="No. of steps to run")
    # parser.add_argument("--sys_name", type=str, default='System', help="System name")
    # parser.add_argument("--traj_folder", type=str, default="/home/civil/phd/cez218288/Benchmarking/MDBENCHGNN/mace_universal_2.0/EXP/Quartz/a.xyz")

    args = parser.parse_args()
    main(args, config)
