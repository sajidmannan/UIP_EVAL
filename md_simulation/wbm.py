from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import time
import os
import torch
import sys
import argparse
from ase.md.nptberendsen import NPTBerendsen
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
from ase import Atoms, units

sys.path.append("/home/civil/phd/cez218288/Benchmarking/MDBENCHGNN/mdbenchgnn/models/mace")
from mace.calculators import MACECalculator
torch.set_default_dtype(torch.float64)

from ase.optimize.optimize import Optimizer
from ase.optimize import FIRE

import json, bz2
from pymatgen.entries.computed_entries import ComputedStructureEntry



def element_to_atomic_number(element_list):
   
    mapping = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 
              'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 
              'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 
              'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 
              'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 
              'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 
              'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 
              'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 
              'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 
              'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 
              'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 
              'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 
              'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 
              'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 
              'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 
              'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 
              'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 
              'Og': 118}

    atomic_numbers_list = [mapping[element] for element in element_list]

    # Convert the list of atomic numbers to a tensor
    atomic_numbers_tensor = torch.tensor(atomic_numbers_list)

    return atomic_numbers_tensor

def map_pbc_to_binary(pbc_list):
   
    binary_list = [1 if value else 0 for value in pbc_list]
    return binary_list
  
  
def convert_to_ase(data):
    # Extract data from the input dictionary
    positions = data['pos'].numpy()
    cell = data['cell'].numpy()
    atomic_numbers = data['atomic_numbers'].numpy()
    pbc = data['pbc'].numpy().astype(bool)

    # Create an ASE Atoms object
    atoms = Atoms(positions=positions,
                    numbers=atomic_numbers,
                    cell=cell,
                    pbc=pbc)

    return atoms


def FIRE_Relax(atoms, calculator, NStep=1e4, ftol=1e-5,plot=False):
    atoms.set_calculator(calculator)
    optimizer = FIRE(atoms,logfile=None)
    trajectory = []
    energies = []
    def record_trajectory():
        trajectory.append(atoms.copy())
        energies.append(atoms.get_potential_energy())
    optimizer.attach(record_trajectory, interval=1)
    optimizer.run(fmax=ftol,steps=NStep)
    
    if(plot):
        # Plotting energy vs. optimization step
        plt.figure()
        plt.plot(energies)
        plt.xlabel('Optimization Step')
        plt.ylabel('Energy (eV)')
        plt.title('Energy vs. Optimization Step')
        plt.grid(True)
        plt.show()
    return trajectory, energies


# Initialize lists for storing predictions and actual values
def initialize_prediction_lists():
    return {
        'Predictions_e': [],
        'Actuals_e': [],
        'formation_pred': [],
        'd_hull_pred': [],
        'Pred_Energy': []
    }

# Function to process a data loader
def process_data_loader(data, sum_data, limit=None):
    calculator = MACECalculator(model_path=args.model_path, device=args.device, default_dtype='float64')

    results = initialize_prediction_lists()
    struc_id = 0

    # Add new columns to sum_data
    sum_data['formation_pred'] = pd.NA
    sum_data['Pred_Energy'] = pd.NA
    sum_data['d_hull_pred'] = pd.NA
    if limit==None:
        limit=len(data['entries'])

    for k in tqdm(range(limit)):
        d = data['entries'][k]['structure']['sites']
        positions_list = []
        atomic_numbers_list = []

        for site in d:
            xyz = site['xyz']
            species = site['species'][0]  
            element = species["element"]  
            positions_list.append(xyz)
            atomic_numbers_list.append(element)
            
        positions_tensor = torch.tensor(positions_list)
        atomic_numbers_tensor = element_to_atomic_number(atomic_numbers_list)

        # Construct the final dictionary
        result_dict = {
            'pos': positions_tensor,
            'cell': torch.tensor(data['entries'][k]['structure']["lattice"]["matrix"]),  
            'atomic_numbers': atomic_numbers_tensor,
            'energy': data['entries'][k]["energy"],  
            'force': torch.zeros((len(d), 3)),  # Assuming a 3D force vector for each site
            'pbc': torch.tensor([1,1,1])  
        }

        # Convert to ASE and perform relaxation (assuming these functions are defined elsewhere)
        atoms = convert_to_ase(result_dict)
        # batch = convAtomstoBatch(atoms)
        OptimTraj, OptimEnergies = FIRE_Relax(atoms, calculator, NStep=args.FIRE_steps, ftol=1e-5, plot=False)

        Pred_Energy = OptimEnergies[-1] 
        formation_pred = Pred_Energy - sum_data["total energy"].iloc[struc_id] + sum_data["formation energy"].iloc[struc_id]
        d_hull_pred = formation_pred - sum_data["d_hull"].iloc[struc_id]

        struc_id += 1
        Actual_Energy = result_dict['energy']
        
        results['Predictions_e'].append(Pred_Energy)
        results['Actuals_e'].append(Actual_Energy)
        results['formation_pred'].append(formation_pred)
        results['d_hull_pred'].append(d_hull_pred)
        results['Pred_Energy'].append(Pred_Energy)

        # Update the sum_data DataFrame
        sum_data.at[struc_id - 1, 'formation_pred'] = formation_pred
        sum_data.at[struc_id - 1, 'Pred_Energy'] = Pred_Energy
        sum_data.at[struc_id - 1, 'd_hull_pred'] = d_hull_pred
        
            
    return results



def main(args):
    with bz2.open(args.data_path) as fh:
        data = json.loads(fh.read().decode('utf-8'))
        

    col=["composition","n_sites","volume","total energy","formation energy","d_hull","band_gap","id"]
    sum_data=pd.read_csv(args.data_summary, header=None)
    sum_data.columns = col
    
    results = process_data_loader(data, sum_data)
    sum_data.to_csv(args.out_path, index=False)

    
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MD simulation with MACE model")
    parser.add_argument("--model_path", type=str, default="example/lips20/botnet/swa_model.pth", help="Path to the model")
    parser.add_argument("--device", type=str, default="cpu", help="Device:['cpu','cuda']")
    parser.add_argument("--data_path", type=str, default="out_dir_sl/neqip/lips20/", help="input data path")
    parser.add_argument("--data_summary", type=str, default="out_dir_sl/neqip/lips20/", help="data summary path")

    parser.add_argument("--out_path", type=str, default="/home/civil/phd/cez218288/Benchmarking/MDBENCHGNN/mace_universal_2.0/WBM/updated_step_1.csv", help="Output path")
    parser.add_argument("--FIRE_steps", type=int, default=10, help="optimization steps")


    args = parser.parse_args()
    main(args)