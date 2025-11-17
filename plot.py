import numpy as np
import h5py

from pathlib import Path
from mosgim2.plot.plot import plot1l, plot2l, plot_1layer_separate_frames, plot_2layer_separate_frames
from converter import prepare_maps, MosgimLayer

def plot(
        mosgim_file: str | Path, 
        plot_files: str | Path, 
        separate_frames: bool,
        maps: dict[str, list] = None
) -> None:
    animation_file = plot_files["animation"]
    data = h5py.File(mosgim_file, 'r')

    IPPh_layer1 = data.attrs['layer1_height']
    ts = data['timestamps']

    # prepare net to estimate TEC on it
    colat = np.arange(2.5, 180, 2.5)
    lon = np.arange(-180, 180, 5.)

    frames1 = maps[MosgimLayer.gim] # for single layer model layer 1 is global map

    if maps[MosgimLayer.plasmasphere]:
        IPPh_layer2 = data.attrs['layer2_height']
        frames1 = maps[MosgimLayer.ionosphere]
        frames2 = maps[MosgimLayer.plasmasphere]
        if separate_frames:
            plot_2layer_separate_frames(plot_files, colat, lon, ts, frames1, frames2, IPPh_layer1 / 1000., IPPh_layer2 / 1000.)
        else:
            plot2l(animation_file, colat, lon, ts, frames1, frames2)
    else:
        if separate_frames:
            plot_1layer_separate_frames(plot_files, colat, lon, ts, frames1, IPPh_layer1 / 1000.)
        else:
            plot1l(animation_file, colat, lon, ts, frames1)   


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot GIMs ')
    parser.add_argument('--in_file',
                        type=Path,
                        help='Path to data, map creation')
    parser.add_argument('--out_file',
                        type=Path,
                        help='Path to video', 
                        default='animation.gif')
    parser.add_argument('--separate_frames',
                        help='If presented separate maps will be saved', 
                        action="store_true")
    args = parser.parse_args()

    maps, _ = prepare_maps(args.in_file)
    if args.separate_frames:
        template = args.out_file.stem
        frame_files = [args.out_file.parent / (template + str(i).zfill(2) + ".png") for i in range(len(maps[MosgimLayer.gim]))]
        files = {'animation': args.out_file, 'frames': frame_files}
    else:
        files = {'animation': args.out_file}
    plot(mosgim_file=args.in_file, plot_files=files, separate_frames=args.separate_frames, maps=maps)



  
