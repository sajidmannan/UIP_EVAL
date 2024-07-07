#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import annotations

import os
import sys
import time
import copy
import yaml
import logging
import argparse
from ase import Atoms
from ase.io import read
from ase.geometry.analysis import Analysis
import random

import numpy as np
import torch
import pytest

from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.singlepoint import SinglePointCalculator as sp

from tqdm import tqdm
import pytorch_lightning as pl

from matsciml.common.registry import registry
from matsciml.common.utils import radius_graph_pbc, setup_imports, setup_logging
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    FrameAveraging,
)
from matsciml.datasets.trajectory_lmdb import data_list_collater
from matsciml.lightning import MatSciMLDataModule
from matsciml.models.pyg import FAENet
from matsciml.models.base import ForceRegressionTask
from matsciml.models.utils.io import multitask_from_checkpoint
from matsciml.preprocessing.atoms_to_graphs import *
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write
import numpy as np
import time
import torch
import argparse
import sys
from tqdm import tqdm
from ase.io import read
from ase.neighborlist import neighbor_list

import os
import time
from matsciml.models.utils.io import * 
import numpy as np
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import matplotlib.pyplot as plt
from ase.io.formats import UnknownFileTypeError
from ase.optimize import FIRE

#sys.path.append("/home/civil/phd/cez218288/Benchmarking/MDBENCHGNN/mdbenchgnn/models/mace")
from mace.calculators import MACECalculator
#torch.set_default_dtype(torch.float64)


# In[7]:


a2g=AtomsToGraphs(max_neigh=200,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=True,
            r_fixed=True,)
f_avg=FrameAveraging(frame_averaging="3D", fa_method="stochastic")

def convAtomstoBatch(atoms):
    data_obj=a2g.convert(atoms)
    Reformatted_batch={
        'cell' : data_obj.cell,
        'natoms' :  torch.Tensor([data_obj.natoms]).unsqueeze(0),
        'edge_index' : [data_obj.edge_index.shape],
        'cell_offsets': data_obj.cell_offsets,
        'y' : None,
        'force' : None, 
        'fixed' : [data_obj.fixed],
        'tags' : None,
        'sid' :None,
        'fid' : None,
        'dataset' : 'S2EFDataset',
        'graph' : data_list_collater([data_obj]),
    }
    Reformatted_batch=f_avg(Reformatted_batch)
    return Reformatted_batch

def convBatchtoAtoms(batch):
    # data_obj=a2g.convert(atoms)
    curr_atoms = Atoms(
            positions=batch['graph'].pos,
            cell = batch['cell'][0],
            numbers=batch['graph'].atomic_numbers,
            pbc=True) # True or false
    
    return curr_atoms


# In[26]:


class TestArgs:
    runsteps=20000
    model_path="/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/2024-01-07-mace-128-L2_epoch-199.model"#"/home/m3rg2000/Simulation/checkpoints-2024/2023-08-14-mace-universal (1).model"
    timestep=0.05
    temp=298
    out_dir="/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/Stability/OutputTestingStability/"
    device='cuda'
    replicate=True
    max_atoms=200  #Replicate upto max_atoms (Min. will be max_atoms/2) (#Won't reduce if more than max_atoms)
    energy_tolerence=0.1
    energy_criteria_interval=10
    max_linedmann_coefficient=0.1
    lindemann_criteria_interval=100
    lindemann_traj_length=100
    max_rdf_error_percent=10
    max_bond_error_percent=10
    bond_criteria_interval=100
    rdf_dr=0.01
    rdf_r_max=6.0
    rdf_traj_length=100
    rdf_criteria_interval=100
    trajdump_interval=10
    minimize_steps=200
args=TestArgs()


# In[22]:


def minimize_structure(atoms, fmax=0.05, steps=50):
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

def min_height(cell_matrix):
    """
    Calculate the perpendicular heights in three directions given a 3x3 cell matrix.
    """
    a,b,c = cell_matrix[:, 0],cell_matrix[:, 1],cell_matrix[:, 2]
    volume = abs(np.dot(a, np.cross(b, c)))
    # Calculate the cross products
    a_cross_b,b_cross_c,c_cross_a = np.linalg.norm(np.cross(a, b)),np.linalg.norm(np.cross(b, c)),np.linalg.norm(np.cross(c, a))
    # Calculate the perpendicular heights
    height_a,height_b,height_c = abs(volume / a_cross_b), abs(volume / b_cross_c),abs(volume / c_cross_a)
    return min(height_a, height_b, height_c)

def perturb_config(atoms,displacement_std=0.01):
    # Create a new Atoms object with the perturbed positions
    positions = atoms.get_positions()
    displacements = np.random.normal(scale=displacement_std, size=positions.shape)
    new_positions = positions + displacements
    new_perturbed_atoms = atoms.copy()
    new_perturbed_atoms.set_positions(new_positions)
    return new_perturbed_atoms

def plot_pair_rdfs(Pair_rdfs,shift=0):
    counter=0
    for key in Pair_rdfs.keys():
        plt.plot(Pair_rdfs[key][0],Pair_rdfs[key][1]+shift*counter,label=key)
        counter+=1
    plt.legend(loc=(1.2,0))
    plt.xlabel("r (Angstrom)")
    plt.ylabel("g(r)")
    plt.show()
    
def replicate_system(atoms, replicate_factors):
    """
    Replicates the given ASE Atoms object according to the specified replication factors.
    """
    nx, ny, nz = replicate_factors
    original_cell = atoms.get_cell()
    original_positions = atoms.get_positions()@original_cell  #Scaled or Unscaled ?
    original_numbers = atoms.get_atomic_numbers()
    x_cell,y_cell,z_cell=original_cell[0],original_cell[1],original_cell[2]
    new_positions, new_numbers = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                new_numbers+=[original_numbers]
    pos_after_x=np.concatenate([original_positions+i*x_cell for i in range(nx)])
    pos_after_y=np.concatenate([pos_after_x+i*y_cell for i in range(ny)])
    pos_after_z=np.concatenate([pos_after_y+i*z_cell for i in range(nz)])
    new_cell = [nx * original_cell[0],ny * original_cell[1],nz * original_cell[2]]
    new_atoms = Atoms(numbers=np.concatenate(new_numbers), positions=pos_after_z, cell=new_cell, pbc=atoms.get_pbc())
    return new_atoms

def write_xyz(Filepath,atoms):
    '''Writes ovito xyz file'''
    R=atoms.get_position()
    species=atoms.get_atomic_numbers()
    cell=atoms.get_cell()
    f=open(Filepath,'w')
    f.write(str(R.shape[0])+"\n")
    flat_cell=cell.flatten()
    f.write(f"Lattice=\"{flat_cell[0]} {flat_cell[1]} {flat_cell[2]} {flat_cell[3]} {flat_cell[4]} {flat_cell[5]} {flat_cell[6]} {flat_cell[7]} {flat_cell[8]}\" Properties=species:S:1:pos:R:3 Time=0.0")
    for i in range(R.shape[0]):
        f.write("\n"+str(species[i])+"\t"+str(R[i,0])+"\t"+str(R[i,1])+"\t"+str(R[i,2]))
        
def Symmetricize_replicate(curr_atoms, max_atoms, box_lengths):
    replication=[1,1,1]
    atom_count=curr_atoms
    lengths=box_lengths
    while atom_count<(max_atoms//2):
        direction=np.argmin(box_lengths)
        replication[direction]+=1
        lengths[direction]=box_lengths[direction]*replication[direction]
        atom_count=curr_atoms*replication[0]*replication[1]*replication[2]
    return replication,atom_count

def get_pairs(atoms):
    Atom_types=np.unique(atoms.get_chemical_symbols())
    Pairs=[]
    for i in range(len(Atom_types)):
        for j in range(i,len(Atom_types)):
            Pairs+=[[Atom_types[i],Atom_types[j]]]   
    return Pairs

def getfirstpeaklength(r,rdf, r_max=6.0):
    bin_size=(r[-1]-r[0])/len(r)
    cut_index=int(r_max/bin_size)
    cut_index=min(cut_index,len(r))
    Peak_index=np.argmax(rdf[:cut_index])
    #Returns : Peak index and Bond length
    return Peak_index , r[Peak_index] 

def get_partial_rdfs(Traj,r_max=6.0):
    rmax=min(r_max,min_height(Traj[0].get_cell())/2.7)
    analysis = Analysis(Traj)
    dr=args.rdf_dr
    nbins=int(rmax/dr)
    pairs_list=get_pairs(Traj[0])
    Pair_rdfs=dict()
    for pair in pairs_list:
        rdf = analysis.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=pair, return_dists=True)
        x=rdf[0][1]
        y=np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
        Pair_rdfs['-'.join(pair)]=[x,y]
    return Pair_rdfs

def get_partial_rdfs_smoothened(inp_atoms,perturb=10,noise_std=0.01,max_atoms=300,r_max=6.0):
    atoms=inp_atoms.copy()
    replication_factors,size=Symmetricize_replicate(len(atoms), max_atoms=max_atoms, box_lengths=atoms.get_cell_lengths_and_angles()[:3])
    atoms=replicate_system(atoms,replication_factors)
    Traj=[perturb_config(atoms,noise_std) for k in range(perturb)]
    return get_partial_rdfs(Traj,r_max=r_max)

def get_bond_lengths_noise(inp_atoms,perturb=10,noise_std=0.01,max_atoms=300,r_max=6.0):
    Pair_rdfs=get_partial_rdfs_smoothened(inp_atoms,perturb=perturb,noise_std=noise_std,max_atoms=max_atoms,r_max=r_max)
    Bond_lengths=dict()
    for key in Pair_rdfs:
        r,rdf=Pair_rdfs[key]
        Bond_lengths[key]=getfirstpeaklength(r,rdf)[1]
    return Bond_lengths

def get_bond_lengths_TrajAvg(Traj,r_max=6.0):
    Pair_rdfs=get_partial_rdfs(Traj,r_max=r_max)
    Bond_lengths=dict()
    for key in Pair_rdfs:
        r,rdf=Pair_rdfs[key]
        Bond_lengths[key]=getfirstpeaklength(r,rdf)[1]
    return Bond_lengths


def get_initial_rdf(inp_atoms,perturb=10,noise_std=0.01,max_atoms=300,replicate=False,Structid=0,r_max=6.0):
    atoms=inp_atoms.copy()
    #write_xyz(f"StabilityXYZData2/{Structid}.xyz",atoms.get_positions(),atoms.get_chemical_symbols(),atoms.get_cell())
    if replicate:
        n_atoms=len(atoms)
        replication_factors,size=Symmetricize_replicate(len(atoms), max_atoms=max_atoms, box_lengths=atoms.get_cell_lengths_and_angles()[:3])
        atoms=replicate_system(atoms,replication_factors)
    rmax=min(r_max,min_height(atoms.get_cell())/2.7)
    #atoms.rattle(0.01)
    analysis = Analysis([perturb_config(atoms,noise_std) for k in range(perturb)])
    #write_xyz(f"StabilityXYZDataReplicated2/{Structid}.xyz",atoms.get_positions(),atoms.get_chemical_symbols(),atoms.get_cell())
    dr=args.rdf_dr
    nbins=int(rmax/dr)
    rdf = analysis.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=None, return_dists=True)
    x=rdf[0][1]
    y=np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
    return x,y

def get_rdf(Traj,r_max=6.0):
    rmax=min(r_max,min_height(Traj[0].get_cell())/2.7)
    analysis = Analysis(Traj)
    dr=args.rdf_dr
    nbins=int(rmax/dr)
    rdf = analysis.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=None, return_dists=True)
    x=rdf[0][1]
    y=np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
    return x,y


# In[28]:


class StabilityException(Exception):
    pass

def run_simulation(atoms,runsteps=1000,timestep=1.0,temp=298,TrajDir='./'):
    start_time = time.time()
    traj = []
    
    _,initial_rdf = get_initial_rdf(atoms,perturb=10,noise_std=0.01,max_atoms=args.max_atoms,replicate=True)
    initial_bond_lengths= get_bond_lengths_noise(atoms,perturb=10,noise_std=0.01,max_atoms=args.max_atoms,r_max=args.rdf_r_max)
    
    #Replicate_system
    replication_factors,size=Symmetricize_replicate(len(atoms), max_atoms=args.max_atoms, box_lengths=atoms.get_cell_lengths_and_angles()[:3])
    atoms=replicate_system(atoms,replication_factors)

    #Set_calculator
    calculator = MACECalculator(model_path=args.model_path, device=args.device, default_dtype='float64')
    atoms.set_calculator(calculator)
    atoms=minimize_structure(atoms,steps=args.minimize_steps)
    #Set_simulation
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    initial_energy = atoms.get_total_energy()
    dyn = VelocityVerlet(atoms, dt=timestep*units.fs)
    
    def write_frame(a=atoms):
        if TrajDir!=None:
            a.write(os.path.join(TrajDir,f'MD_{atoms.get_chemical_formula()}_NVE.xyz'), append=True)
        traj.append(a.copy())

    dyn.attach(write_frame, interval=args.trajdump_interval)
    
    def energy_criterion(atoms, initial_energy, tolerance=0.10):
        current_energy = atoms.get_total_energy()
        energy_error = abs((current_energy-initial_energy)/initial_energy)
        return energy_error<tolerance

    def energy_stability(a=atoms):
        if not energy_criterion(a, initial_energy, tolerance=args.energy_tolerence):
            raise StabilityException("Energy_criterion violated. Stopping the simulation.")

    dyn.attach(energy_stability, interval=args.energy_criteria_interval)   ### energy dumping interval

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
        i, j, _ = neighbor_list('ijd', atoms, cutoff=5.0)  ## Cutoff change
        distances = atoms.get_distances(i, j, mic=True)
        return np.mean(distances)

    def lindemann_stability(a=atoms):
        if len(traj) >= args.lindemann_traj_length:
            rmsd = calculate_rmsd(traj[-args.lindemann_traj_length:])
            avg_nn_distance = calculate_average_nn_distance(traj[0])
            lindemann_coefficient = np.mean(rmsd) / avg_nn_distance
            if lindemann_coefficient > args.max_linedmann_coefficient:
                raise StabilityException(f"lindemann_stability criterion violated{lindemann_coefficient}>{args.max_linedmann_coefficient}, Stopping the simulation.")

    dyn.attach(lindemann_stability, interval=args.lindemann_criteria_interval)   ### last 1000 frames msd

    def rdf_stability(a=atoms):
        if len(traj) >= args.rdf_traj_length:
            _,rdf = get_rdf(traj[-args.rdf_traj_length:],r_max=args.rdf_r_max)
            
            error_rdf=100*(((rdf-initial_rdf)**2).sum())/(((initial_rdf)**2).sum())
            if error_rdf > args.max_rdf_error_percent:
                raise StabilityException("RDF criterion violated. Stopping the simulation. WF="+str(error_rdf))    
    
    dyn.attach(rdf_stability, interval=args.rdf_criteria_interval)   ### last 1000 frames msd  

    #Lindemann Bond lengths
    def bond_lengths_stability(a=atoms):
        if len(traj) >= args.lindemann_traj_length:
            curr_bond_lengths=get_bond_lengths_TrajAvg(traj[-args.rdf_traj_length:],r_max=args.rdf_r_max)
            for key in curr_bond_lengths.keys():
                Error_percent=100*abs((curr_bond_lengths[key]-initial_bond_lengths[key]))/initial_bond_lengths[key]
                if(Error_percent > args.max_bond_error_percent):
                    raise StabilityException(f"Bond length stability violated. Stopping the simulation. Bond {key}={curr_bond_lengths[key]} , Initial={initial_bond_lengths[key]}")
    dyn.attach(bond_lengths_stability, interval=args.rdf_criteria_interval)   ### last 1000 frames msd  

    try:
        print(f"Simulating {atoms.get_chemical_formula()} {len(atoms)} atoms system ....")
        counter=0
        for k in tqdm(range(runsteps)):
            dyn.run(1) 
            counter+=1
        return runsteps  # Simulation completed successfully
    except StabilityException as e:
        print(f"Simulation of {atoms.get_chemical_formula()} {len(atoms)} atoms system failed after {counter} steps")
        # print(f"File: {os.path.basename(cif_file_path)} - {e}")
        return len(traj)  # Return the number of steps completed before failure


# In[24]:

# Seed for the Python random module
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)  # if you are using multi-GPU.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

print("All seeds set!")



# Load Data
dm = MatSciMLDataModule(
    "MaterialsProjectDataset",
    train_path="/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/Stability/stability_new/",#"/home/m3rg2000/matsciml/Scale_new_lmdb/stability_new",#TRAIN_PATH,
    # val_split=VAL_PATH,
    # test_split=VAL_PATH,
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(cutoff_radius=6.0, adaptive_cutoff=True),
            PointCloudToGraphTransform(
                "pyg",
                node_keys=["pos", "atomic_numbers"],
            ),
            #FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
        ],
    },
    batch_size=1,
)

dm.setup()
train_loader = dm.train_dataloader()
dataset_iter = iter(train_loader)


# Main execution
# cif_folder = "/home/m3rg2000/Simulation/checkpoints-2024/"
# failed_simulations = []

time_steps = []
unreadable_files = []



Range=[0,120]
TrajDir=os.path.join(args.out_dir, f'TrajDir_{Range[0]}_{Range[1]}')
os.makedirs(TrajDir, exist_ok=True)
counter_batch=0
for batch in train_loader:
    if counter_batch<Range[1] and counter_batch>=Range[0]:
        atoms=convBatchtoAtoms(batch)
        steps_completed = run_simulation(atoms,args.runsteps,args.timestep,args.temp,TrajDir)        
        time_steps.append(steps_completed)
        print("System: ",counter_batch,f" : {atoms.get_chemical_formula()} with {len(atoms)} atoms stopped at ",time_steps," steps")
        counter_batch+=1
    else:
        counter_batch+=1
        continue
#
print("Completed...")
print("Time Steps:",time_steps)

# # In[10]:


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run MD simulation with MACE model")
    
#     parser.add_argument("--model_path", type=str, default="/home/m3rg2000/Simulation/checkpoints-2024/FAENet_250k.ckpt", help="Path to the model")
#     parser.add_argument("--device", type=str, default="cpu", help="Device:['cpu','cuda']")
#     parser.add_argument("--out_dir", type=str, default="/home/m3rg2000/Simulation/OutputTestingStability", help="Output path")
#     parser.add_argument("--temp", type=float, default=298, help="Temperature in Kelvin")
#     parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs units")
#     parser.add_argument("--runsteps", type=int, default=1000, help="No. of steps to run")
#     parser.add_argument("--sys_name", type=str, default='System', help="System name")
#     parser.add_argument("--energy_criteria_interval", type=int, default=10, help="Energy Criteria Interval")
#     parser.add_argument("--replicate", type=bool, default=True, help="Replicate the system")
#     parser.add_argument("--max_atoms", type=int, default=200, help="Max atoms (Min. will be max_atoms/2)")
#     parser.add_argument("--energy_tolerance", type=float, default=0.1, help="Energy tolerance")
#     parser.add_argument("--max_lindemann_coefficient", type=float, default=0.1, help="Max Lindemann coefficient")
#     parser.add_argument("--lindemann_criteria_interval", type=int, default=50, help="Lindemann criteria interval")
#     parser.add_argument("--lindemann_traj_length", type=int, default=50, help="Lindemann trajectory length")
#     parser.add_argument("--max_rdf_error_percent", type=float, default=10, help="Max RDF error percent")
#     parser.add_argument("--max_bond_error_percent", type=float, default=10, help="Max bond error percent")
#     parser.add_argument("--bond_criteria_interval", type=int, default=100, help="Bond criteria interval")
#     parser.add_argument("--rdf_dr", type=float, default=0.01, help="RDF dr")
#     parser.add_argument("--rdf_r_max", type=float, default=6.0, help="RDF r max")
#     parser.add_argument("--rdf_traj_length", type=int, default=500, help="RDF trajectory length")
#     parser.add_argument("--rdf_criteria_interval", type=int, default=100, help="RDF criteria interval")
#     parser.add_argument("--trajdump_interval", type=int, default=1, help="Trajectory dump interval")
    
#     args = parser.parse_args()
#     main(args)

