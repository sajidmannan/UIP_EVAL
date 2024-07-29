###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model for 
# energy and force MAE
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

#import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, utils, torch_tools
""" python /home/civil/phd/cez218288/scratch/mace_v_0.3.5/md_simulation/mace/eval_mae.py --configs "/home/civil/phd/cez218288/Benchmarking/MDBENCHGNN/example/lips_1/data/test/botnet.xyz" --model "/scratch/scai/phd/aiz238703/MDBENCHGNN/Repulsive/OutputZBL1/MACE_model_500_lips_ZBL1_swa.model"  --device cuda"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    # parser.add_argument("--output",help="output path",required=True)
    parser.add_argument("--device",help="select device",type=str,choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--default_dtype",help="set default dtype",type=str,choices=["float32", "float64"],default="float64")
    parser.add_argument("--batch_size", help="batch size", type=int, default=1)
    parser.add_argument("--info_prefix",help="prefix for energy, forces and stress keys",type=str,default="MACE_")
    
    
    return parser.parse_args()



import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def plot_r2_score(actual, pred, title="Title"):
    save_dir = './'  # Ensure this path ends with a slash

    # Calculate R² score
    r2 = r2_score(actual, pred)

    # Create the scatter plot
    plt.scatter(actual, pred)
    plt.xlabel("Actual", fontsize=20, fontweight='bold')
    plt.ylabel("Predicted", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')

    # Plot the 45-degree line
    min_val = min(min(actual), min(pred))
    max_val = max(max(actual), max(pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    plt.title(title, fontsize=20, fontweight='bold')

    # Annotate the R² score on the plot
    plt.text(0.05, 0.95, f'R² = {r2:.5f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Save the plot to the specified location with the title in the filename
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(f'{save_dir}{filename}')

    # Show the plot
    plt.show()
    # Clear the current figure to avoid overlap
    plt.clf()



def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device).to(device)
    model=model.double()

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max))
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    
    #Create counter variables
    counter=0
    e_mae=0
    f_mae=0
    e_rmse=0
    f_rmse=0
        
    Predictions_Fx=[]
    Actuals_Fx=[]

    Predictions_Fy=[]
    Actuals_Fy=[]

    Predictions_Fz=[]
    Actuals_Fz=[]
    for batch in data_loader:
        counter+=1
        batch = batch.to(device)
        output = model(batch.to_dict())
        temp_e1=abs(batch['energy']).mean()
        temp_f1=abs(batch['forces']).mean()

        
        temp_e=(abs(batch['energy']-output['energy'])).mean()
        temp_f=(abs(batch['forces']-output['forces'])).mean()
        temp_re=torch.sqrt(((batch['energy']-output['energy'])**2).mean())
                          
        temp_rf=torch.sqrt(((batch['forces']-output['forces'])**2).mean())
        Pred_Forces = output['forces']
        Actual_Forces = batch['forces']
        
        Predictions_Fx+=Pred_Forces[:,0].reshape(-1).detach().cpu().numpy().tolist()
        Actuals_Fx+=Actual_Forces[:,0].reshape(-1).detach().cpu().numpy().tolist()

        Predictions_Fy+=Pred_Forces[:,1].reshape(-1).detach().cpu().numpy().tolist()
        Actuals_Fy+=Actual_Forces[:,1].reshape(-1).detach().cpu().numpy().tolist()

        Predictions_Fz+=Pred_Forces[:,2].reshape(-1).detach().cpu().numpy().tolist()
        Actuals_Fz+=Actual_Forces[:,2].reshape(-1).detach().cpu().numpy().tolist()
        
        counter+=1
        if(counter>500):
            break
        
        # print("Batch: ",counter,"\te_mae: ",round((temp_e-temp_e1).item(),3),"\tf_mae: ",round((temp_f-temp_f1).item(),3))
        print("Batch_old: ",counter,"\te_mae: ",round((temp_e).item(),3),"\tf_mae: ",round((temp_f).item(),3))

        e_mae+=temp_e #-temp_e1
        f_mae+=temp_f #-temp_f1
        
        e_rmse+=temp_re #-temp_e1
        f_rmse+=temp_rf #-temp_f1
        
        
        
    print("||Final Results:||")
    print("E_MAE: ", round((e_mae/counter).item(),3) ,"\t F_MAE: ",round((f_mae/(counter)).item(),3) )
    print("E_RMSE: ", round((e_rmse/counter).item(),3) ,"\t F_RMSE: ",round((f_rmse/(counter)).item(),3) )
    
    
    plot_r2_score(Actuals_Fx, Predictions_Fx, "UpstreamMacelips_Fx")
    plot_r2_score(Actuals_Fy, Predictions_Fy, "UpstreamMacelips_Fy")
    plot_r2_score(Actuals_Fz, Predictions_Fz, "UpstreamMacelips_Fz")
    
if __name__ == "__main__":
    main()
