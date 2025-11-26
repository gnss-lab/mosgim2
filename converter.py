import argparse
import numpy as np
from pathlib import Path
from mosgim2.plot.frames import makeframes
from datetime import datetime, UTC

from ionex_formatter.reader import get_single_map
from ionex_formatter.ionex_map import SpatialRange
from ionex_formatter.formatter import IonexFile, IonexMapType, HeaderConfig
from numpy.typing import NDArray
from typing import Tuple
import config

import h5py

from typing import Tuple
from datetime import datetime, UTC, timedelta
from enum import Enum

description = [
"Global ionosphere maps for day {doy}, {year} ({date})",       
"",                                                           
"Padokhin, A. M., E. S. Andreeva, M. O. Nazarenko, and",
"S. A. Kalashnikova. Phase-Difference Approach for GNSS",
"Global Ionospheric Total Electron Content Mapping.", 
"Radiophysics and Quantum Electronics, 65(7): 481-495,2023",
"https://doi.org/10.1007/s11141-023-10230-6"
"                                                            "
]

comment = [
    "TEC values in  0.1 TECUs; 9999 if no value available        ",
    "IGS GPS stations used in the computations:                  "
]
        
order = [
    "IONEX VERSION / TYPE",
    "PGM / RUN BY / DATE",
    "DESCRIPTION",
    "EPOCH OF FIRST MAP",
    "EPOCH OF LAST MAP",
    "INTERVAL",
    "# OF MAPS IN FILE",
    "MAPPING FUNCTION",
    "ELEVATION CUTOFF",
    "# OF STATIONS",
    "# OF SATELLITES",
    "OBSERVABLES USED",
    "BASE RADIUS",
    "MAP DIMENSION",
    "HGT1 / HGT2 / DHGT",
    "LAT1 / LAT2 / DLAT",
    "LON1 / LON2 / DLON",
    "EXPONENT",
    "COMMENT",
    "START OF AUX DATA",
    "END OF AUX DATA",
    "END OF HEADER"
]


class MosgimProduct(Enum):
    ionex: str = "INX"
    snapshot: str = "png"
    animation: str = "gif"
    coefficients: str = "hdf5"
    observation: str = "h5"

class MosgimLayer(Enum):
    ionosphere: str = "INS"
    plasmasphere: str = "PLS"
    gim: str = "GIM"
    combined: str = "CMB"

class MosgimStages(Enum):
    final: str = "FIN"

def __get_durtation_sampling(duration: timedelta, sampling: timedelta) -> Tuple[str, str]:
    seconds_in_day = 3600 * 24
    seconds_in_hour = 3600
    seconds_in_minute = 60
    duration_seconds = duration.days * seconds_in_day + duration.seconds # microseconds ignored
    if duration_seconds > seconds_in_day:
        raise ValueError("Interval longer than 1 day are not implemented HERE for IONEX names. Read IONEX format")
    if duration_seconds == seconds_in_day:
        duration_ionex = "01D"
    elif duration_seconds % seconds_in_hour == 0:
        nhours = duration_seconds // seconds_in_hour
        duration_ionex = str(nhours).zfill(2) + "H"
    elif duration_seconds < seconds_in_hour:
        raise ValueError("Interval shorter than 1 hour are not implemented HERE for IONEX names. Read IONEX format")
    
    sampling_seconds = sampling.days * seconds_in_day + sampling.seconds # microseconds ignored
    if sampling_seconds < 1 or int(sampling_seconds) != sampling_seconds:
        raise ValueError(
                "Sampling with less than second or with non-integer seconds not" \
                " implemented HERE for IONEX names. Read IONEX format"
            )
    if sampling_seconds < seconds_in_minute:
        sampling_ionex = str(sampling_seconds).zfill(2) + "S"
    elif sampling_seconds < seconds_in_hour:
        nminutes = sampling_seconds // seconds_in_minute
        sampling_ionex = str(nminutes).zfill(2) + "M"
    else:
        if sampling_seconds % seconds_in_hour != 0:
            raise ValueError("Sampling with non-integer hourss are not implemented HERE for IONEX names. Read IONEX format")   
        nhours = sampling_seconds // seconds_in_hour
        sampling_ionex = str(nhours).zfill(2) + "H"
    return duration_ionex, sampling_ionex

def get_ionexlike_fname(
        epoch: datetime, 
        duration: timedelta, 
        product: MosgimProduct, 
        layer: MosgimLayer, 
        version: int, 
        stage:MosgimStages = MosgimStages.final,
        sampling: timedelta = timedelta(seconds=3600),
        specification: str = "OPS"
) -> str:
    if not 0 <= version <= 9:
        raise ValueError("MOSGIM ionex version should be in range from 0 to 9")
    if epoch.tzinfo != UTC:
        raise ValueError("The epoch should have UTC timezone")
    duration_ionex, sampling_ionex = __get_durtation_sampling(duration, sampling)
 
    # The length of each filed matters CCC means 3-character field
    # "{CCC}{V}OPSTYP_{YEAR}{DOY}{HH}{MM}_{DUR}_SMP_CNT.{EXT}"
    template = "{center}{version}{specification}{stage}_{timelabel}_{duration}_{sampling}_{layer}.{extension}"
    fname = template.format(
        center="MOS",
        version=version,
        specification = specification,
        stage=stage.value,
        timelabel = epoch.strftime("%Y%j%H%M"),
        duration=duration_ionex,
        sampling = sampling_ionex,
        layer = layer.value,
        extension = product.value
    )
    return fname


def prepare_maps(mosgim_file: str | Path) -> Tuple[dict[str, list[NDArray]], NDArray]:
    data = h5py.File(mosgim_file, 'r')

    nlayers = data.attrs['nlayers']
    coord =  data.attrs['coord']
    nbig_layer1, mbig_layer1 = data.attrs['layer1_dims'] 
    res_layer1 = data['layer1_SHcoefs'] 
    if nlayers == 2:
        nbig_layer2, mbig_layer2 = data.attrs['layer2_dims'] 
        res_layer2 = data['layer2_SHcoefs']

    ts = data['timestamps']
    # prepare net to estimate TEC on it
    colats = np.arange(2.5, 180, 2.5)
    lats = np.arange(87.5, -90, -2.5)
    lons = np.arange(-180, 180, 5.)
    spatial_grid = np.meshgrid(lons, lats)

    maps = {MosgimLayer.ionosphere: None, MosgimLayer.plasmasphere: None, MosgimLayer.gim: None}
    maps_layer1 = makeframes(lons, colats, coord, nbig_layer1, mbig_layer1, res_layer1, ts)    
    maps_layer1 = [np.array(m) for m in maps_layer1]
    maps[MosgimLayer.gim] = maps_layer1

    if nlayers == 2:
        maps_layer2 = makeframes(lons, colats, coord, nbig_layer2, mbig_layer2, res_layer2, ts)    
        maps_layer2 = [np.array(m) for m in maps_layer2]
        maps[MosgimLayer.gim] = [l1 + l2 for l1, l2 in zip(maps_layer1, maps_layer2)]
        maps[MosgimLayer.ionosphere] = maps_layer1
        maps[MosgimLayer.plasmasphere] = maps_layer2

    return maps, spatial_grid

def save_ionex(
    fpath: Path,
    maps_data: list[NDArray], 
    epochs:list[datetime],
    spatial_grid: NDArray, 
    header: HeaderConfig,
):
    lon_grid = spatial_grid[0]
    lat_grid = spatial_grid[1]
    ionex_maps = {}
    for map_data, epoch in zip(maps_data, epochs):
        _arr = np.array(map_data)
        ionex_maps[epoch] = get_single_map(_arr, epoch, lat_grid, lon_grid)
    formatter = IonexFile()
    formatter.set_maps(ionex_maps, dtype=header.map_type)
    ionex_lines = formatter.get_file_lines(header, saved_types=[header.map_type])
    with open(fpath, "w") as f:
        f.writelines("\n".join(ionex_lines))

def get_epochs(mosgim_file: str | Path) -> datetime:
    data = h5py.File(mosgim_file, 'r')
    ts = data['timestamps']
    epochs = [datetime.fromtimestamp(t, tz=UTC) for t in ts]
    return epochs

def get_heights(mosgim_file: str | Path) -> dict[MosgimLayer, float]:
    heights = {}
    with h5py.File(mosgim_file, 'r') as data:
        heights = {
            MosgimLayer.ionosphere: data.attrs["layer1_height"] / 1000, 
            MosgimLayer.gim: data.attrs["layer1_height"] / 1000
        }
        if data.attrs["nlayers"] == 2:
            heights[MosgimLayer.plasmasphere] = data.attrs["layer2_height"] / 1000
    return heights


def convert(
        mosgim_file: str | Path, 
        ionex_files: dict[str, Path],
        maps: dict[str, list[NDArray]], 
        spatial_grid: NDArray
) -> None:
    with h5py.File(mosgim_file, 'r') as data:
        nmaps = data.attrs['nmaps']    
        sites = data.attrs['sites']
    epochs = get_epochs(mosgim_file)
    heights = get_heights(mosgim_file)
    year = str(epochs[0].year)
    doy = str(epochs[0].timetuple().tm_yday).zfill(3)
    description[0] = description[0].format(doy=doy, year=year, date=epochs[0].strftime("%d-%m-%Y"))

    header_config = HeaderConfig(
        map_type=IonexMapType.TEC,
        pgm = "MOSGIM",
        run_by = "GNSS-Lab",
        created_at = datetime.now(UTC),
        first_time = epochs[0],
        last_time = epochs[-1],
        description = description,
        timestep = epochs[1] - epochs[0],
        number_of_maps=nmaps,
        elevation_cutoff = np.degrees(config.el_cutoff),
        number_of_stations = 300,
        number_of_satellites = 32,
        sites_names = sites,
        version = "1.0",
        gnss_type = "GPS",
        mapping_function = "COSZ", 
        base_radius = 6371.0,
        latitude_range = SpatialRange(87.5, -87.5, -2.5),
        longitude_range = SpatialRange(-180, 180, 5),
        height_range = SpatialRange(450, 450, 0),
        exponent = -1,
        map_dimensions = 2,
        comment = comment,
        labels_order = order
    )
    for layer, layer_maps in maps.items():
        if not layer_maps:
            #files[layer] = None
            continue
        header_config.height_range = SpatialRange(heights[layer], heights[layer], 0)
        save_ionex(ionex_files[layer], layer_maps, epochs, spatial_grid, header_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert GIMs data to IONEX format')
    parser.add_argument('--in_file',
                        type=Path,
                        help='Path to data with calculations result')
    parser.add_argument('--out_path',
                        type=Path,
                        help='Path to IONEX files', default='animation.gif')
    
    args = parser.parse_args()
    epochs = get_epochs(args.in_file)
    ion_ionex = get_ionexlike_fname(epochs[0], timedelta(days=1), MosgimProduct.ionex, MosgimLayer.ionosphere, version=0)
    pla_ionex = get_ionexlike_fname(epochs[0], timedelta(days=1), MosgimProduct.ionex, MosgimLayer.plasmasphere, version=0)
    gim_ionex = get_ionexlike_fname(epochs[0], timedelta(days=1), MosgimProduct.ionex, MosgimLayer.gim, version=0)
    files = {
        MosgimLayer.ionosphere: args.out_path / ion_ionex, 
        MosgimLayer.plasmasphere: args.out_path / pla_ionex, 
        MosgimLayer.gim: args.out_path / gim_ionex
    }
    maps, spatial_grid = prepare_maps(args.in_file)
    convert(args.in_file, files, maps, spatial_grid)