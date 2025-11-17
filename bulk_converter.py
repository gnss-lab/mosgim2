import argparse
import os
import h5py

from pathlib import Path
from datetime import datetime, timedelta, UTC

# imports from project
from converter import MosgimLayer, MosgimProduct, MosgimStages, get_ionexlike_fname, get_epochs
from converter import prepare_maps, convert
from plot import plot as plot_maps




def build_datetime_filepath_dict(root_dir: Path, prefix='MOS0OPSFIN_'):
    """
    Build a dictionary mapping datetime objects to file paths.
    
    Iterates over directory structure: root/YYYY/DDD/MOS0OPSFIN_YYYYDDD0000_01D_01H_CMB.hdf5
    Extracts YYYY and DDD from the filename to construct the datetime.

    """
    files = {}
    
    # Walk through all directories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Only process HDF5 files matching our pattern
            if filename.startswith(prefix) and filename.endswith(".hdf5"):
                try:
                    # Extract the date part from filename
                    # Format: MOS0OPSFIN_YYYYDDD0000_01D_01H_CMB.hdf5
                    timelabel = filename.split('_')[1]
                    dt = datetime.strptime(timelabel, "%Y%j%H%M").replace(tzinfo=UTC)

                    file_path = Path(dirpath) / filename
                    files[dt] = file_path.absolute()
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    files = dict(sorted(files.items()))
    return files

def parse_datetime(datetime_str):
    try:
        epoch = datetime.strptime(datetime_str, "%Y-%m-%d")
        epoch = epoch.replace(tzinfo=UTC)
        return epoch
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format: '{datetime_str}'. Expected YYYY-MM-DD"
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Conversion of MosGIM rtesults to plots and ionex.")
    # Add arguments for each config option
    parser.add_argument("--mosgim_results", type=str, help="Path to results directory")
    parser.add_argument("--start_date", type=parse_datetime, help="Start date for conversion in format %Y-%m-%d")
    parser.add_argument("--end_date", type=parse_datetime, help="End date for conversion in format %Y-%m-%d")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cmd_args = parse_args()
    mosgim_results_path = cmd_args.mosgim_results
    files = build_datetime_filepath_dict(mosgim_results_path)
    print(f"Found {len(files)} files")
    ionex_file = None
    for epoch, mosgim_file in files.items():
        if cmd_args.start_date and epoch < cmd_args.start_date:
            continue
        if cmd_args.end_date and epoch > cmd_args.end_date:
            continue
        plot_files = {'frames': [], 'animation': None}
        try:
            ion_ionex = get_ionexlike_fname(epoch, timedelta(days=1), MosgimProduct.ionex, MosgimLayer.ionosphere, version=0)
            pla_ionex = get_ionexlike_fname(epoch, timedelta(days=1), MosgimProduct.ionex, MosgimLayer.plasmasphere, version=0)
            gim_ionex = get_ionexlike_fname(epoch, timedelta(days=1), MosgimProduct.ionex, MosgimLayer.gim, version=0)
            ionex_files = {
                MosgimLayer.ionosphere: mosgim_file.parent / ion_ionex, 
                MosgimLayer.plasmasphere: mosgim_file.parent / pla_ionex, 
                MosgimLayer.gim: mosgim_file.parent / gim_ionex
            }
            with h5py.File(mosgim_file, 'r') as mosgim_data:
                IPPh_layer1 = mosgim_data.attrs['layer1_height']
                IPPh_layer2 = mosgim_data.attrs['layer2_height']   

            heights = {
                MosgimLayer.ionosphere: IPPh_layer1 / 1000, 
                MosgimLayer.plasmasphere: IPPh_layer2 / 1000, 
                MosgimLayer.gim: IPPh_layer1 / 1000
            }
            maps, spatial_grid = prepare_maps(mosgim_file)
            animation_fname = get_ionexlike_fname(
                epoch, timedelta(days=1), MosgimProduct.animation, MosgimLayer.combined, version=0
            )
            for hour in range(25):
                snapshot_fname = get_ionexlike_fname(
                    epoch+timedelta(hours=hour), 
                    timedelta(hours=1), 
                    MosgimProduct.snapshot, 
                    MosgimLayer.combined, 
                    version=0
                )
                plot_files['frames'].append( mosgim_file.parent / snapshot_fname )
            plot_files['animation'] = mosgim_file.parent / animation_fname
            
            convert(mosgim_file, ionex_files, maps, spatial_grid)
            plot_maps(mosgim_file, plot_files, separate_frames=True, maps=maps)
        except Exception as e:
            print(f"Failed to process {mosgim_file} due to {e}")