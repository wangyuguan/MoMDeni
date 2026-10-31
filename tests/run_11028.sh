#!/usr/bin/env bash
#SBATCH --job-name=kam11028
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --constraint=h100
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --mail-user=yw3384@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=NONE
#SBATCH --chdir=/home/yw3384/softwares/denoised_kam
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

RESULTS=/scratch/gpfs/CRYOEM/yw3384/Kam-cryoEM-FB/results
RUN_ROOT="${RESULTS}/11028_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/slurm.out") 2> >(tee -a "$RUN_ROOT/slurm.err" >&2)

set +u
source /home/yw3384/softwares/anaconda3/etc/profile.d/conda.sh
conda activate /home/yw3384/softwares/anaconda3/envs/aspire
set -u
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 SLURM_EXPORT_ENV=ALL
command -v kam >/dev/null

# MoM initializer: global (common) shift correction only, the remaining shifts are modeled by a shift distribution on rings
# (0.25 px spacing up to a quarter of the radius of the 20x20 moment images) fitted jointly in the M3 stage.
# EMPIAR-11028 (GR-Hsp90-p23, az8940-processed stack): the particles are NOT contrast-inverted (dark particles), hence --no-flip-sign.
kam pipeline /tigress/CRYOEM/singerlab/EMPIAR11028/processed_particles.star \
  --dataset-label EMPIAR-11028 \
  --no-flip-sign \
  --datadir /tigress/CRYOEM/singerlab/EMPIAR11028 \
  --outdir "$RESULTS" \
  --n-select 10000 \
  --denoise-res 64 \
  --denoise-max-order 40 \
  --do-global-centering \
  --vol-degree 7 \
  --viewing-degree 2 \
  --moment-res 20 \
  --augment-reflections \
  --n-init 3 \
  --maxiter 10000 \
  --shift-distribution \
  --shift-step-px 0.25 \
  --device gpu \
  --workers 16 \
  "$@"

for arg in "$@"; do
  case "$arg" in --dry-run|--help|-h) exit 0 ;; esac
done

# Native-grid RELION inputs. NOTE: this script runs the standard protocol only. The paper's EMPIAR-11028 refinement also used a hand-made soft
# solvent mask (--solvent_mask), which kam relion_prepare cannot generate, so these numbers are not expected to reproduce that run exactly.
kam relion_prepare \
  --pipeline-dir "$RUN_ROOT/kam_pipeline" \
  --particles /scratch/gpfs/CRYOEM/yw3384/EMPIAR-11028/processed_particles_nobeamtilt.star \
  --datadir /tigress/CRYOEM/singerlab/EMPIAR11028 \
  --native-pixel-size 0.835 \
  --particle-diameter 160 \
  --symmetry C1 \
  --offset-range 10 \
  --offset-step 2 \
  --low-resol-join-halves 40

# RELION refinement.
# This version of relion_prepare always writes --ini_high (default 30 A); the reported refinements use none.
sed -i 's/ --ini_high [0-9.]*//' "$RUN_ROOT/relion_prepare/run_refinement.sh"
! grep -q -- "--ini_high" "$RUN_ROOT/relion_prepare/run_refinement.sh"

bash "$RUN_ROOT/relion_prepare/run_refinement.sh"

# Masked, sharpened resolution (comparable with published numbers).
set +u
source /etc/profile.d/modules.sh
module purge
module use /projects/CRYOEM/Modules/modulefiles-rhel9
module load relion/5.0.1/gcc-14.2.1-cpu
set -u
kam postprocess \
  --relion-dir "$RUN_ROOT/relion_refine" \
  --molecular-weight 2.63e5

# Optional reference comparison.
kam align_reference \
  --relion-dir "$RUN_ROOT/relion_refine" \
  --reference-map /home/yw3384/softwares/Kam-cryoEM/reference_map/emd_23004.map
