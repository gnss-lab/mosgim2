import h5py
import os
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from config_simurg import CANDIDATE_TEMPLATE, MOSGIM_FILES_ROOT, OBSERVATION_FILES_TEMPLATE
from converter import MosgimStages

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("MOSGIM-Site-Extractor")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.propagate = False

SITES = [
    'abpo', 'albh', 'algo', 'alic', 'amu2', 'antc', 'areg', 'areq', 'ascg', 'auck', 'badg', 'bake', 
    'bako', 'barh', 'bjfs', 'bndy', 'bogo', 'bogt', 'bor1', 'braz', 'brft', 'brst', 'brux', 'bshm', 
    'cags', 'cas1', 'ccj2', 'cedu', 'chpg', 'chpi', 'chti', 'chur', 'ckis', 'coco', 'cord', 'cpvg', 
    'cro1', 'daej', 'darw', 'dav1', 'dhlg', 'drao', 'dubo', 'ebre', 'faa1', 'fair', 'flin', 'flrs', 
    'frdn', 'func', 'ganp', 'glps', 'glsv', 'gode', 'godz', 'gol2', 'gold', 'gope', 'gras', 'graz', 
    'guat', 'guug', 'harb', 'hers', 'hert', 'hlfx', 'hnlc', 'hnpt', 'hob2', 'hofn', 'holb', 'hyde', 
    'ieng', 'iisc', 'invk', 'iqal', 'irkj', 'joz2', 'joze', 'karr', 'kerg', 'kir0', 'kiru', 'kit3', 
    'kokb', 'kokv', 'koug', 'kour', 'krgg', 'laut', 'lmmf', 'lpgs', 'm0se', 'mac1', 'madr', 'mal2', 
    'mana', 'mas1', 'mat1', 'mate', 'maui', 'maw1', 'mcil', 'mcm4', 'mdo1', 'mdvj', 'medi', 'mets', 
    'mgue', 'mizu', 'mkea', 'mobj', 'mobs', 'monp', 'morp', 'mqzg', 'mro1', 'mtka', 'nano', 'nico', 
    'nist', 'nklg', 'nlib', 'nnor', 'not1', 'nrc1', 'ntus', 'nya1', 'nyal', 'ohi2', 'onsa', 'ous2', 
    'pado', 'parc', 'park', 'pdel', 'penc', 'pert', 'pie1', 'pimo', 'pohn', 'pol2', 'polv', 'pots', 
    'pove', 'prds', 'ptbb', 'ptvl', 'qaq1', 'rabt', 'ramo', 'reun', 'reyk', 'rgdg', 'rio2', 'riop', 
    'salu', 'samo', 'sant', 'savo', 'sch2', 'scor', 'scrz', 'sctb', 'scub', 'seyg', 'smst', 'sofi', 
    'spt0', 'ssia', 'sthl', 'stjo', 'stk2', 'str2', 'suth', 'sutm', 'suwn', 'sydn', 'syog', 'tabl', 
    'tabv', 'tash', 'tehn', 'thtg', 'thti', 'thu2', 'tid1', 'tidb', 'tixg', 'tlse', 'tnml', 'tong', 
    'tow2', 'tro1', 'tsea', 'tskb', 'tuva', 'twtf', 'uclu', 'ufpr', 'ulab', 'unb3', 'unbj', 'unbn', 
    'unsa', 'urum', 'usn7', 'usud', 'vald', 'vill', 'voim', 'wab2', 'wark', 'wes2', 'wgtn', 'whit', 
    'will', 'wind', 'wsrt', 'wtza', 'wtzl', 'wtzr', 'wtzs', 'wtzz', 'wuhn', 'xmis', 'yakt', 'yar2', 
    'yar3', 'yarr', 'yell', 'ykro', 'yssk', 'zamb', 'zeck', 'zim2', 'zim3', 'zimm'
]

def copy_hdf5_group(src_group, dest_group):
    """
    Recursively copy contents (datasets, groups, attributes)
    from one HDF5 group to another.
    """
    # Copy group attributes
    for key, value in src_group.attrs.items():
        dest_group.attrs[key] = value

    # Copy datasets and subgroups
    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            new_group = dest_group.create_group(name)
            copy_hdf5_group(item, new_group)
        elif isinstance(item, h5py.Dataset):
            if not name in ["tec", "azimuth", "elevation", "timestamp"]:
                continue
            dest_group.create_dataset(name, data=item[()])
            # Copy dataset attributes if any
            for key, value in item.attrs.items():
                dest_group[name].attrs[key] = value


def compare_files(original_file: Path, copied_file: Path, sites: list[str]) -> list[str]:
    """
    Compare two HDF5 files and return a list of site names that:
      - Are present in the original file,
      - Are in the 'sites' argument list,
      - But are missing from the copied file.

    Args:
        original_file (Path): Path to the original HDF5 file.
        copied_file (Path): Path to the copied (partial) HDF5 file.
        sites (list[str]): List of site names to check.

    Returns:
        list[str]: Site names that are in the original file but absent in the copied file.
    """
    if not original_file.exists():
        raise FileNotFoundError(f"Original file not found: {original_file}")
    if not copied_file.exists():
        raise FileNotFoundError(f"Copied file not found: {copied_file}")

    missing_sites = []

    with h5py.File(original_file, "r") as orig, h5py.File(copied_file, "r") as copy:
        for site in sites:
            if site in orig and site not in copy:
                missing_sites.append(site)

    return missing_sites

def extract_sites_from_hdf(source_path: str, target_path: str, site_list: list[str]):
    """
    Extract specified sites (and their contents + attributes)
    from an HDF5 file and save into a new HDF5 file.

    Args:
        source_path (str): Path to the original HDF5 file.
        target_path (str): Path to the new filtered HDF5 file.
        site_list (list[str]): List of site names to extract.

    Returns:
        str: Path to the new filtered HDF5 file.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
    start = time.time()
    with h5py.File(source_path, "r") as src, h5py.File(target_path, "w") as dest:
        for site_name in site_list:
            if site_name in src:
                src_site_group = src[site_name]
                dest_site_group = dest.create_group(site_name)
                copy_hdf5_group(src_site_group, dest_site_group)
            else:
                logger.info(f"⚠️ Site '{site_name}' not found in source file.")
    logger.info(f"Elapsed time for {target_path} is {time.time() - start}")
    return target_path


def parse_args():
    parser = argparse.ArgumentParser(description="Parse args for sites extractor.")

    # Add arguments for each config option
    parser.add_argument("--source_path", type=Path, help="Path to hdf5 big file", default=None)
    parser.add_argument("--target_path", type=Path, help="Path to hdf5 containing only selected sites", default=None)
    parser.add_argument("--start_date", type=str, help="Start date to extract data in YYYY-MM-DD format", default=None)
    parser.add_argument("--skip_existing", type=bool, help="Checkput file storage to retrive last date", default=True)
    parser.add_argument("--stage", type=MosgimStages, help="Defines stage for which data are extracted", required=True)
    args = parser.parse_args()
    if args.start_date:
        args.start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    return args

def get_files(start: datetime, end: datetime):
    result = {}
    current  = start
    while current <= end:
        year = current.year
        doy = str(current.timetuple().tm_yday).zfill(3)
        candidates = [c.format(year=year, doy=doy) for c in CANDIDATE_TEMPLATE]
        for f in candidates:
            print(f)
            if os.path.exists(f):
                result[current] = f
                break
        current = current + timedelta(days=1)
    return result

def process_file(in_file: Path, out_file: Path, sites: list[str]=SITES):
    status = "extract"
    if not Path(in_file).exists():
        status = "SOURCE-FILE-MISSING"
    if Path(in_file).exists() and Path(out_file).exists():
        missing = compare_files(Path(in_file), Path(out_file), sites)
        if  len(missing) > 5:
            msg = f"Target file {out_file} exists but several sites are missed: {missing}" \
                    "To keep observation consistent with GIMS, data will NOT be overwritten" \
                    "Consider to use different calculation compaing for example RAP -> FIN"
            logger.warning(msg)
        status = "FILE-EXIST"
    if status == "extract":
        out_file.parent.mkdir(exist_ok=True, parents=True)
        extract_sites_from_hdf(in_file, out_file, sites)
        logger.info(f"Extracted data from {in_file} to {out_file}")
    else:
        logger.info(f"Skip extraction from {in_file} to {out_file} since: {status}")
        return None
    return out_file

def verify_extraction(target_file: Path) -> bool:
    """Verifies hdf file is valid 
    """
    with h5py.File(target_file, "r") as target:
        for site_name in target:
            # Verification is made by opening file and readine site
            # TODO make verification more reliable if needed
            continue
    return True


def process_range(
        out_root_path: Path, 
        start: datetime,
        end: datetime,
        skip_existing: bool,
        stage: MosgimStages
) -> list[Path]:
    files = get_files(start, end)
    if not files:
        logger.warning(f"No files from {start} to {end}")
    extracted = list()
    for date, in_file in files.items():

        logger.info(f"Process {date}: {in_file}")
        year = str(date.year)
        doy = str(date.timetuple().tm_yday).zfill(3)
        fname = OBSERVATION_FILES_TEMPLATE[stage].format(year=year, doy=doy)
        out_file = out_root_path / year / doy / fname
        if skip_existing and out_file.exists():
            logger.info(f"Skip existing file {out_file}")
            continue
        for attempt in range(3): # at most three attempts to extrac file, idealy one is enough
            out_file = process_file(in_file, out_file, SITES)
            if out_file:
                try:
                    verify_extraction(out_file)
                    extracted.append(out_file)
                    break
                except Exception as e:
                    logger.exception(f"Attempt {attempt} to extract from {in_file} to {out_file} failed during verification")
                    continue
    return extracted


if __name__ == "__main__":
    args = parse_args()
    if not (args.source_path is None) and not (args.target_path is None):
        if not args.start_date is None:
            logger.warning(f"Option --start_date is ignored since source and target are given")
        if args.skip_existing is True and args.target_path.exists():
            logger.error(f"Option --skip_existing is True, make it False if what override")
            exit
        logger.info(f"Extract from {args.source_path} to {args.target_path}")
        files = process_file(args.source_path, args.target_path)

    elif not args.start_date is None:
        logger.info(f"Assume source data in {CANDIDATE_TEMPLATE} and result are in {MOSGIM_FILES_ROOT}")
        files = process_range(
            out_root_path = MOSGIM_FILES_ROOT,
            start = args.start_date, 
            end = datetime.now() - timedelta(days=3),
            skip_existing = args.skip_existing,
            stage=args.stage
        )
    else:
        logger.warning("Define start_date or source_path/target_path for script to work")
