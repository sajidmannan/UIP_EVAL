from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import time
import os
import torch
import sys
import argparse
from ase.optimize import BFGS  # Import BFGS optimizer
sys.path.append("/home/civil/phd/cez218288/Benchmarking/MDBENCHGNN/mdbenchgnn/models/mace")  # Path to mace directory relative to MDBENCHGNN Folder
from mace.calculators import MACECalculator
from tqdm import tqdm
torch.set_default_dtype(torch.float64)
from ase.md.nptberendsen import NPTBerendsen
from ase import Atoms
from ase.optimize import FIRE

def get_density(atoms):
    amu_to_grams = 1.66053906660e-24  # 1 amu = 1.66053906660e-24 grams
    angstrom_to_cm = 1e-8  # 1 Å = 1e-8 cm
    mass_amu = atoms.get_masses().sum()
    mass_g = mass_amu * amu_to_grams    # Get the volume of the atoms object in cubic angstroms (Å³)
    volume_A3 = atoms.get_volume()
    volume_cm3 = volume_A3 * (angstrom_to_cm ** 3)  # 1 Å³ = 1e-24 cm³
    density = mass_g / volume_cm3

    return density

def write_xyz(Filepath, atoms):
    '''Writes ovito xyz file'''
    R = atoms.get_positions()
    species = atoms.get_atomic_numbers()
    cell = atoms.get_cell()
    
    with open(Filepath, 'w') as f:
        f.write(str(R.shape[0]) + "\n")
        flat_cell = cell.flatten()
        f.write(f"Lattice=\"{flat_cell[0]} {flat_cell[1]} {flat_cell[2]} {flat_cell[3]} {flat_cell[4]} {flat_cell[5]} {flat_cell[6]} {flat_cell[7]} {flat_cell[8]}\" Properties=species:S:1:pos:R:3 Time=0.0")
        for i in range(R.shape[0]):
            f.write("\n" + str(species[i]) + "\t" + str(R[i, 0]) + "\t" + str(R[i, 1]) + "\t" + str(R[i, 2]))

def replicate_system(atoms, replicate_factors):
    """
    Replicates the given ASE Atoms object according to the specified replication factors.

    Parameters:
    atoms (ase.Atoms): The original atoms object to be replicated.
    replicate_factors (tuple): A tuple of three integers (nx, ny, nz) specifying the 
                               replication factors in the x, y, and z directions.

    Returns:
    ase.Atoms: A new Atoms object that is the replicated version of the input atoms object.
    """
    nx, ny, nz = replicate_factors
    original_cell = atoms.get_cell()
    original_positions = atoms.get_scaled_positions() @ original_cell
    original_numbers = atoms.get_atomic_numbers()
    x_cell, y_cell, z_cell = original_cell[0], original_cell[1], original_cell[2]
    new_positions = []
    new_numbers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                new_numbers += [original_numbers]
    pos_after_x = [original_positions + i * x_cell for i in range(nx)]
    pos_after_x = np.concatenate(pos_after_x)
    pos_after_y = [pos_after_x + i * y_cell for i in range(ny)]
    pos_after_y = np.concatenate(pos_after_y)
    pos_after_z = [pos_after_y + i * z_cell for i in range(nz)]
    pos_after_z = np.concatenate(pos_after_z)
    # Create new cell
    new_cell = [
        nx * original_cell[0],
        ny * original_cell[1],
        nz * original_cell[2]
    ]
    # Create new Atoms object
    new_atoms = Atoms(numbers=np.concatenate(new_numbers), positions=pos_after_z, cell=new_cell, pbc=atoms.get_pbc())
    return new_atoms



def minimize_structure(atoms, fmax=0.05, steps=200):
    """
    Perform energy minimization on the given ASE Atoms object using the FIRE optimizer.

    Parameters:
    atoms (ase.Atoms): The Atoms object to be minimized.
    fmax (float): The maximum force tolerance for the optimization (default: 0.01 eV/Å).
    steps (int): The maximum number of optimization steps (default: 1000).

    Returns:
    ase.Atoms: The minimized Atoms object.
    """
    dyn = FIRE(atoms, trajectory=None)
    dyn.run(fmax=fmax, steps=steps)
    return atoms


def run_simulation(calculator, atoms, pressure=0.000101325, temperature=298, timestep=0.1, steps=10, TrajPath='xyz'):  ##pressure=0.000101325 GPa
         
    # Define the temperature and pressure
    init_conf = atoms
    init_conf.set_calculator(calculator)
    # Initialize the NPT dynamics
    dyn = NPTBerendsen(init_conf, timestep=timestep * units.fs, temperature_K=temperature,
                       pressure_au=pressure * units.bar * 1e4, compressibility_au=4.57e-5 / units.bar)
    
    density = []
    angles = []
    lattice_parameters = []
    def write_frame():
        dyn.atoms.write(TrajPath, append=True)
        cell = dyn.atoms.get_cell()
        
        lattice_parameters.append(cell.lengths())  # Get the lattice parameters
        angles.append(cell.angles())  # Get the angles
        density.append(get_density(atoms))  
    
    dyn.attach(write_frame, interval=10)
    dyn.run(steps)

    density = np.array(density)
    angles = np.array(angles)
    lattice_parameters = np.array(lattice_parameters)
    
    # Calculate average values
    avg_density = np.mean(density)
    avg_angles = np.mean(angles, axis=0)
    avg_lattice_parameters = np.mean(lattice_parameters, axis=0)    
    return avg_density, avg_angles, avg_lattice_parameters

def main(args):
    start_time = time.time() 
    calculator = MACECalculator(model_path=args.model_path, device=args.device, default_dtype='float64')
    calculator.model.double()  # Change model weights type to double precision (hack to avoid error)
    model_name = "mace"  
    cif_files_dir = args.input_dir
    output_file = args.out_dir
    import pandas as pd
    from tqdm import tqdm
    # List to hold the data
    data = []
    for folder in tqdm(os.listdir(cif_files_dir)):
        folder_path = os.path.join(cif_files_dir, folder)
        
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                Temp,Press=file.split("_")[2:4]
                Temp,Press=float(Temp),float(Press)
                TrajPath=os.path.join(args.traj_folder,"_".join(file.split("_")[:2])+'_Trajectory.xyz')  
                
                try:
                    atoms = read(file_path)

                    # Replicate the system if needed
                    n_atoms = len(atoms)  # Uncomment if you want to replicate the system
                    n_repeat = int(np.ceil((100 / n_atoms) ** (1 / 3)))
                    atoms = replicate_system(atoms, (n_repeat, n_repeat, n_repeat))

                    # Minimize the structure
                    atoms.set_calculator(calculator)
                    atoms = minimize_structure(atoms)

                    # Calculate density and cell lengths and angles
                    density = get_density(atoms)
                    cell_lengths_and_angles = atoms.get_cell_lengths_and_angles().tolist()

                    avg_density, avg_angles, avg_lattice_parameters = run_simulation(calculator, atoms, pressure=Press, temperature=Temp, timestep=args.timestep, steps=args.runsteps, TrajPath=TrajPath)
                    print(avg_density)
                    # Append the results to the data list
                    data.append([folder + file[:-4], density] + cell_lengths_and_angles + [avg_density] + avg_lattice_parameters.tolist() + avg_angles.tolist())
                     # Create a DataFrame
                    columns = ["Filename", "Exp_Density (g/cm³)", "Exp_a (Å)", "Exp_b (Å)", "Exp_c (Å)", "Exp_alpha (°)", "Exp_beta (°)", "Exp_gamma (°)"
                            ,"Sim_Density (g/cm³)", "Sim_a (Å)", "Sim_b (Å)", "Sim_c (Å)", "Sim_alpha (°)", "Sim_beta (°)", "Sim_gamma (°)"]
                    df = pd.DataFrame(data, columns=columns)
        
                    # Save the DataFrame to a CSV file
                    df.to_csv(output_file, index=False)
                    print(f"Data saved to {output_file}")
                except:
                    print("filename", file_path)                    

    # Create a DataFrame
    # columns = ["Filename", "Exp_Density (g/cm³)", "Exp_a (Å)", "Exp_b (Å)", "Exp_c (Å)", "Exp_alpha (°)", "Exp_beta (°)", "Exp_gamma (°)"
    #            ,"Sim_Density (g/cm³)", "Sim_a (Å)", "Sim_b (Å)", "Sim_c (Å)", "Sim_alpha (°)", "Sim_beta (°)", "Sim_gamma (°)"]
    # df = pd.DataFrame(data, columns=columns)
    
    # # Save the DataFrame to a CSV file
    # df.to_csv(output_file, index=False)
    # print(f"Data saved to {output_file}")

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Run MD simulation with MACE model")
    parser.add_argument("--model_path", type=str, default="example/lips20/botnet/swa_model.pth", help="Path to the model")
    parser.add_argument("--init_conf_path", type=str, default="example/lips20/data/test/botnet.xyz", help="Path to the initial configuration")
    parser.add_argument("--device", type=str, default="cuda", help="Device: ['cpu', 'cuda']")
    parser.add_argument("--input_dir", type=str, default="./", help="folder path")
    parser.add_argument("--out_dir", type=str, default="out_dir_sl/neqip/lips20/exp.csv", help="Output path")
    parser.add_argument("--temp", type=float, default=300, help="Temperature in Kelvin")
    parser.add_argument("--pressure", type=float, default=1, help="pressure in atm")
    parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs units")
    parser.add_argument("--runsteps", type=int, default=1000, help="No. of steps to run")
    parser.add_argument("--sys_name", type=str, default='System', help="System name")
    parser.add_argument("--traj_folder", type=str, default="/home/civil/phd/cez218288/Benchmarking/MDBENCHGNN/mace_universal_2.0/EXP/Quartz/a.xyz")

    args = parser.parse_args()
    main(args) 
