#!/usr/bin/env bash
#SBATCH --job-name=kam10180
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --constraint=h100
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --mail-user=yw3384@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=NONE
#SBATCH --chdir=/home/yw3384/softwares/Kam-cryoEM-FB
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

RESULTS=/scratch/gpfs/CRYOEM/yw3384/Kam-cryoEM-FB/results
RUN_ROOT="${RESULTS}/10180_${SLURM_JOB_ID}"
DATA=/tigress/CRYOEM/singerlab/mg6942/empiar10180_data/data
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/slurm.out") 2> >(tee -a "$RUN_ROOT/slurm.err" >&2)

set +u
source /home/yw3384/softwares/anaconda3/etc/profile.d/conda.sh
conda activate /home/yw3384/softwares/anaconda3/envs/aspire
set -u
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 SLURM_EXPORT_ENV=ALL
command -v kam >/dev/null

# MoM initializer.
kam pipeline "$DATA/Example/consensus_data.star" \
  --dataset-label EMPIAR-10180 \
  --datadir "$DATA" \
  --outdir "$RESULTS" \
  --n-select 10000 \
  --denoise-res 64 \
  --centering \
  --vol-degree 7 \
  --viewing-degree 2 \
  --moment-res 20 \
  --augment-reflections \
  --n-init 3 \
  --device gpu \
  --workers 16 \
  "$@"

for arg in "$@"; do
  case "$arg" in --dry-run|--help|-h) exit 0 ;; esac
done

# Reference for the comparison below: no deposited map for this dataset is
# mirrored locally and Mask-and-Ref holds only multi-body bodies and masks, so
# the reference is the mean of the two consensus half-maps shipped with the
# particles. It is a reconstruction of these same images, not an independent
# structure, so agreement with it is weaker evidence than for the other sets.
REFERENCE="$RUN_ROOT/consensus_mean.mrc"
python - "$REFERENCE" "$DATA/Example/consensus_half1_class001.mrc" \
    "$DATA/Example/consensus_half2_class001.mrc" << 'PYTHON'
import sys

import mrcfile
import numpy as np

target, *halves = sys.argv[1:]
volumes, voxel = [], None
for path in halves:
    with mrcfile.open(path) as handle:
        volumes.append(handle.data.astype(np.float32))
        voxel = float(handle.voxel_size.x)
        if volumes[-1].shape != volumes[0].shape:
            raise SystemExit("consensus half-maps disagree in shape")
with mrcfile.new(target, overwrite=True) as handle:
    handle.set_data(sum(volumes) / len(volumes))
    handle.voxel_size = voxel
print(f"wrote {target}: {volumes[0].shape[0]} pixels at {voxel:.4f} A")
PYTHON

# Native-grid RELION inputs and initial low-pass. The particle diameter follows
# the consensus refinement that produced those half-maps, and the offset search
# is wider than for the other datasets because the origins here have RMS 40 and
# 99th percentile 86 native pixels.
kam relion_prepare \
  --pipeline-dir "$RUN_ROOT/kam_pipeline" \
  --particle-diameter 473 \
  --symmetry C1 \
  --offset-range 50 \
  --offset-step 10 \
  --low-resol-join-halves 40

# RELION refinement. 327k particles at 320 pixels, so this is the longest of
# the four runs; the walltime above is set accordingly.
# This version of relion_prepare always writes --ini_high (default 30 A); the reported refinements use none.
sed -i 's/ --ini_high [0-9.]*//' "$RUN_ROOT/relion_prepare/run_refinement.sh"
! grep -q -- "--ini_high" "$RUN_ROOT/relion_prepare/run_refinement.sh"

bash "$RUN_ROOT/relion_prepare/run_refinement.sh"

# Masked, sharpened resolution (comparable with published numbers). The mass is
# that of the pre-catalytic spliceosome B complex and only sets the solvent
# mask volume.
set +u
source /etc/profile.d/modules.sh
module purge
module use /projects/CRYOEM/Modules/modulefiles-rhel9
module load relion/5.0.1/gcc-14.2.1-cpu
set -u
kam postprocess \
  --relion-dir "$RUN_ROOT/relion_refine" \
  --molecular-weight 1.8e6

# Optional reference comparison.
kam align_reference \
  --relion-dir "$RUN_ROOT/relion_refine" \
  --reference-map "$REFERENCE"
