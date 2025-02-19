import csv
import os
import uuid
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm
from utils import *


def process_slice(args):
    root_folder, model_name, slice_number, xyz_files_slice, log_files_slice = args
    unique_uuid = uuid.uuid1().__str__()
    file_handler = FileHandler(root_folder, incoming_uuid=unique_uuid)
    calculator = PropertyCalculator()

    master_densities = []
    master_lattice_params = []
    master_temperature = []
    master_rdf_values = []
    master_time_temp_data = []
    os.makedirs(f"./results/{model_name}/slice_{slice_number}", exist_ok=True)
    for (system_name, xyz_file_path), (_, log_file_path) in tqdm(
        zip(xyz_files_slice, log_files_slice),
        total=len(xyz_files_slice),
        desc=f"Processing Slice {slice_number}",
    ):
        (
            densities,
            lattice_params,
            temperature,
            rdf_error,
            time_temp_data,
            bond_error,
        ) = process_file(
            file_handler, calculator, system_name, xyz_file_path, log_file_path, model_name
        )

        bond_error_file_name = (
            f"./results/{model_name}/slice_{slice_number}/bond_errors_{model_name}.txt"
        )
        save_bond_errors_to_txt(bond_error_file_name, bond_error)

        master_densities.append(densities)
        master_lattice_params.append(lattice_params)
        master_temperature.append(temperature)
        master_rdf_values.append(rdf_error)
        master_time_temp_data.append(time_temp_data)

    os.makedirs(f"./results/{model_name}/slice_{slice_number}", exist_ok=True)

    save_to_csv(
        f"./results/{model_name}/slice_{slice_number}/master_densities_{model_name}.csv",
        master_densities,
    )
    save_to_csv(
        f"./results/{model_name}/slice_{slice_number}/master_lattice_params_{model_name}.csv",
        master_lattice_params,
    )
    save_to_csv(
        f"./results/{model_name}/slice_{slice_number}/master_temperature_{model_name}.csv",
        [[temp] for temp in master_temperature],
    )
    save_to_csv(
        f"./results/{model_name}/slice_{slice_number}/master_rdf_values_{model_name}.csv",
        master_rdf_values,
    )
    save_to_csv(
        f"./results/{model_name}/slice_{slice_number}/master_time_temp_data_{model_name}.csv",
        master_time_temp_data,
    )
    print(f"Data saved for slice {slice_number} with model name '{model_name}'.")


def main(root_folder, model_name, index):
    file_handler = FileHandler(root_folder, incoming_uuid=uuid.uuid1().__str__())
    xyz_files, log_files = file_handler.find_xyz_files()

    total_files = len(xyz_files)
    num_slices = 600
    slice_size = total_files // num_slices

    args_list = [
        root_folder,
        model_name,
        index,
        xyz_files[index * slice_size : (index + 1) * slice_size],
        log_files[index * slice_size : (index + 1) * slice_size],
    ]

    process_slice(args_list)


# Example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument('--root', type=str, required=True)
    args = parser.parse_args()
    index = args.index
    root_folder = args.root
    model_name = Path(root_folder).name
    main(root_folder, model_name, index)
