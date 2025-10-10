import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import h5py

# imports from project
from mosgim2.data.loader import LoaderTxt, LoaderHDF
from mosgim2.data.tec_prepare import(process_data, combine_data,
                                     calc_coordinates,
                                     get_data,
                                     sites)

import config
from mosgim2.mosgim.mosgim import solve_all as solve1
from mosgim2.mosgim.mosgim2 import solve_all as solve2
from mosgim2.consts.phys_consts import secs_in_day, POLE_THETA, POLE_PHI
from mosgim2.data.writer import writer

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Parse MosGIM configuration.")

    # Add arguments for each config option
    parser.add_argument("--data_path", type=str, help="Path to data directory")
    parser.add_argument("--res_path", type=str, help="Path to results directory")
    parser.add_argument("--nworkers", type=int, help="Number of CPU cores to use")
    parser.add_argument("--coords", type=str, choices=["mag", "geo", "modip"], help="Type of coordinates to use")
    parser.add_argument("--data_type", type=str, choices=["tecsuite-dat", "simurg-hdf"], help="Type of input data")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cmd_args = parse_args()

    res_path = cmd_args.res_path if cmd_args.res_path else config.res_path 
    data_path = cmd_args.data_path if cmd_args.data_path else config.data_path 
    coords = cmd_args.coords if cmd_args.coords else config.coords
    nworkers = cmd_args.nworkers if cmd_args.nworkers else config.nworkers
    data_type = cmd_args.data_type if cmd_args.data_type else config.data_type
    if data_path == "/PATH/TO/INPUT/DATA":
        raise ValueError("Specify path to input data")
    if res_path == "/PATH/TO/RESULTS":
        raise ValueError("Specify path to results")
    IPPh_layer1 = config.IPPh_layer1
    IPPh_layer2 = config.IPPh_layer2
    nbig_layer1 = config.nbig_layer1
    mbig_layer1 = config.mbig_layer1  
    nbig_layer2 = config.nbig_layer2
    mbig_layer2 = config.mbig_layer2
    tint = config.tint 
    sigma0 = config.sigma0
    sigma_v = config.sigma_v
    linear = config.linear
    lcp = config.lcp 
    nlayers = config.nlayers
    maxgap = config.maxgap
    maxjump = config.maxjump 
    el_cutoff = config.el_cutoff
    derivative = config.derivative 
    short = config.short 
    sparse = config.sparse

    n_coefs_layer1 = (nbig_layer1 + 1)**2 - (nbig_layer1 - mbig_layer1) * (nbig_layer1 - mbig_layer1 + 1)
    n_coefs_layer2 = (nbig_layer2 + 1)**2 - (nbig_layer2 - mbig_layer2) * (nbig_layer2 - mbig_layer2 + 1)
    n_coefs = n_coefs_layer1 + n_coefs_layer2

    st = time.time()

    if data_type == "tecsuite-dat" and Path(data_path).is_dir(): 
        loader = LoaderTxt(root_dir=data_path, IPPh1 = IPPh_layer1, IPPh2 = IPPh_layer2)
    elif data_type == "simurg-hdf" and Path(data_path).is_file() and data_path.endswith(".h5"):
        loader = LoaderHDF(data_path, IPPh1 = IPPh_layer1 / 1000., IPPh2 = IPPh_layer2 / 1000.)
    else:
        raise ValueError(
            f"Could not handle the input data {data_path} for type {data_type}." \
            " Must be folder with subfolders with dat-files or h5-file.")
    data_generator = loader.generate_data(sites=sites)

    data = process_data(data_generator, maxgap = maxgap, maxjump = maxjump, el_cutoff = el_cutoff,
                        derivative = derivative, short = short, sparse = sparse)
    print(sorted(set(sites) - set(loader.not_found_sites)))
    data_combined = combine_data(data)
    result = calc_coordinates(data_combined, coords)
    data, time0 = get_data(result)

    print(f'Preprocessing done, took {time.time() - st}')

    st = time.time()

    ndays = np.ceil((np.max(data['time']) - np.min(data['time'])) / secs_in_day).astype('int') # number of days in input file

    nT_add = 1 if linear else 0

    if not (ndays == 1 or ndays == 3):
        print('procedure only works with 1 or 3 consecutive days data')
        exit(1)

    if nlayers == 2:
        res, disp_scale, Ninv = solve2(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, IPPh_layer1, IPPh_layer2, tint, sigma0, sigma_v, data, gigs=2, lcp=lcp, nworkers=nworkers, linear=linear)
    if nlayers == 1:
        res, disp_scale, Ninv = solve1(nbig_layer1, mbig_layer1, IPPh_layer1, tint, sigma0, sigma_v, data, gigs=2, lcp=lcp, nworkers=nworkers, linear=linear)

    if ndays == 3:
        res = res[n_coefs * (tint):n_coefs * (2 * tint + nT_add)] # select central day from 3day interval
        time0 = time0 + timedelta(days=1) 

    print(f'Computation done, took {time.time() - st}')

    res_file = Path(res_path, time0.strftime("%Y-%m-%d")+'.hdf5')

    if nlayers == 1:
        if coords == 'mag':
            writer(filename=res_file, res=res, time0=time0, nmaps=tint + nT_add, linear=linear, coord=coords, 
                   nlayers=nlayers, layer1_dims=[nbig_layer1, mbig_layer1], layer1_height=IPPh_layer1, 
                   sites=sorted(set(sites) - set(loader.not_found_sites)), pole_colat = POLE_THETA, pole_long = POLE_PHI)
        else:
            writer(filename=res_file, res=res, time0=time0, nmaps=tint + nT_add, linear=linear, coord=coords, 
                   nlayers=nlayers, layer1_dims=[nbig_layer1, mbig_layer1], layer1_height=IPPh_layer1, 
                   sites=sorted(set(sites) - set(loader.not_found_sites)))
  
    if nlayers == 2:
        if coords == 'mag':        
            writer(filename=res_file, res=res, time0=time0, nmaps=tint + nT_add, linear=linear, coord=coords, 
                   nlayers=nlayers, layer1_dims=[nbig_layer1, mbig_layer1], layer1_height=IPPh_layer1, layer2_dims = [nbig_layer2, mbig_layer2], layer2_height = IPPh_layer2, 
                   sites=sorted(set(sites) - set(loader.not_found_sites)), pole_colat = POLE_THETA, pole_long = POLE_PHI)  
        else:
            writer(filename=res_file, res=res, time0=time0, nmaps=tint + nT_add, linear=linear, coord=coords, 
                   nlayers=nlayers, layer1_dims=[nbig_layer1, mbig_layer1], layer1_height=IPPh_layer1, layer2_dims = [nbig_layer2, mbig_layer2], layer2_height = IPPh_layer2, 
                   sites=sorted(set(sites) - set(loader.not_found_sites)))  

