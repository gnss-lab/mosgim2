from pathlib import Path
from converter import MosgimStages

MOSGIM_FILES_ROOT = Path("/tmp/mosgim/root")
SIMURG_FILES_ROOTS = [
    Path("/tmp/simurg/root1"),
    Path("/tmp/simurg/root2")
]

OBSERVATION_FILES_TEMPLATE = {
    MosgimStages.preliminary: "MOS0OPSPRE_{year}{doy}0000_01D_30S_OBS.h5",
    MosgimStages.final: "MOS0OPSFIN_{year}{doy}0000_01D_30S_OBS.h5",
    MosgimStages.rapid: "MOS0OPSRAP_{year}{doy}0000_01D_30S_OBS.h5",
}

CANDIDATE_TEMPLATE = [ str(froot / "{year}/{doy}/out/data.h5") for froot in SIMURG_FILES_ROOTS]

if not MOSGIM_FILES_ROOT.exists():
    raise RuntimeError(f"Define correct path to MOSGIM data, {MOSGIM_FILES_ROOT} not exists")

for simurg_root in SIMURG_FILES_ROOTS:
    if not simurg_root.exists():
        raise RuntimeError(f"Define correct path to SIMURG data, {simurg_root} not exists")