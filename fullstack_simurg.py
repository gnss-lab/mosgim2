import argparse
import requests
import os

from pathlib import Path
from datetime import datetime, timedelta, UTC
from typing import Dict

# imports from project
from process import process
from converter import MosgimLayer, MosgimProduct, get_ionexlike_fname
from converter import prepare_maps, convert
from plot import plot as plot_maps

def parse_datetime(datetime_str):
    try:
        epoch = datetime.strptime(datetime_str, "%Y-%m-%d")
        epoch = epoch.replace(tzinfo=UTC)
        return epoch
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format: '{datetime_str}'. Expected YYYY-MM-DD"
        )
   
def get_simurg_hdf(epoch: datetime, working_dir: Path) -> Path:
    str_date = epoch.strftime("%Y-%m-%d")
    url = f"https://simurg.space/gen_file?data=obs&date={str_date}"
    print(f"Loading data from {url}...")
    local_file = working_dir / f"observations_{str_date}.h5"
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
    if  local_file.exists() and 'Content-Length' in response.headers:
        local_size = os.path.getsize(local_file) 
        remote_size = int(response.headers['Content-Length'])
        if local_size == remote_size:
            print("File already exists and is of the same size")
            return local_file

    with open(local_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"File '{local_file}' downloaded successfully.")

    return local_file

def convert_and_plot(mosgim_file: Path, epoch: datetime) -> Dict[MosgimProduct | str, Path]:
    try:
        ion_ionex = get_ionexlike_fname(epoch, timedelta(days=1), MosgimProduct.ionex, MosgimLayer.ionosphere, version=0)
        pla_ionex = get_ionexlike_fname(epoch, timedelta(days=1), MosgimProduct.ionex, MosgimLayer.plasmasphere, version=0)
        gim_ionex = get_ionexlike_fname(epoch, timedelta(days=1), MosgimProduct.ionex, MosgimLayer.gim, version=0)
        ionex_files = {
            MosgimLayer.ionosphere: mosgim_file.parent / ion_ionex, 
            MosgimLayer.plasmasphere: mosgim_file.parent / pla_ionex, 
            MosgimLayer.gim: mosgim_file.parent / gim_ionex
        }
        plot_files = {'frames': [], 'animation': None}
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

        maps, spatial_grid = prepare_maps(mosgim_file)        
        convert(mosgim_file, ionex_files, maps, spatial_grid)
        plot_maps(mosgim_file, plot_files, separate_frames=True, maps=maps)
    except Exception as e:
        print(f"Failed to process {mosgim_file} due to {e}")
    files = {
        MosgimProduct.ionex: ionex_files,
        MosgimProduct.animation: plot_files['animation'],
        MosgimProduct.snapshot: plot_files['frames']
    }
    return files


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


def full_stack(epoch: datetime, working_dir: Path, nworkers: int, coords: str):
    try:
        observation_file = get_simurg_hdf(epoch, working_dir)
        mosgim_file = get_ionexlike_fname(epoch, timedelta(days=1), MosgimProduct.coefficients, MosgimLayer.combined, version=0)
        mosgim_file = working_dir / mosgim_file
        observation_epoch = process(observation_file, mosgim_file, "simurg-hdf", coords, nworkers)  
        if observation_epoch and observation_epoch != epoch: 
            raise ValueError(
                f"Something wrong with data. Data are retrieved for {epoch}." \
                f" However timestamps in data indentify {observation_epoch}"
            )
        print(f"Saved MOSGIM results in '{mosgim_file}' ")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred when retrieved data from remote server (SIMuRG): {e}")
    except Exception as e:
        print(f"Failed to process {epoch} due to:\n" + str(e))

    files = convert_and_plot(mosgim_file, epoch)
    files[MosgimProduct.coefficients] = mosgim_file 
    return files

def parse_args():
    parser = argparse.ArgumentParser(description="Full stack (calculation, ionex, plots) MosGIM configuration.")

    # Add arguments for each config option
    parser.add_argument("--working_dir", type=Path, help="Path to calculation results")
    parser.add_argument("--date", type=parse_datetime, help="Start date for conversion in format %Y-%m-%d")
    parser.add_argument("--nworkers", type=int, help="Number of CPU cores to use")
    parser.add_argument("--coords", type=str, choices=["mag", "geo", "modip"], help="Type of coordinates to use")
    parser.add_argument("--lag_days", type=int, default=5, help="Number of days in past to check observation")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cmd_args = parse_args()
    
    working_dir = cmd_args.working_dir
    coords = cmd_args.coords 
    epoch = cmd_args.date
    nworkers = cmd_args.nworkers 
    full_stack(epoch, working_dir, nworkers, coords)
