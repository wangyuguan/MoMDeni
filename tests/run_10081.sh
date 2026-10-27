#!/usr/bin/env bash
#SBATCH --job-name=kam10081
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
RUN_ROOT="${RESULTS}/10081_${SLURM_JOB_ID}"
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
kam pipeline /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/empiar_10081_hcn1/particles.star \
  --dataset-label EMPIAR-10081 \
  --datadir /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/empiar_10081_hcn1 \
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

# Native-grid RELION inputs (unmasked, unfiltered initializer; no RELION initial low-pass).
kam relion_prepare \
  --pipeline-dir "$RUN_ROOT/kam_pipeline" \
  --particle-diameter 180 \
  --symmetry C1 \
  --offset-range 10 \
  --offset-step 2 \
  --low-resol-join-halves 40

# RELION refinement.
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
  --molecular-weight 4.5e5

# Optional reference comparison.
kam align_reference \
  --relion-dir "$RUN_ROOT/relion_refine" \
  --reference-map /home/yw3384/softwares/Kam-cryoEM/reference_map/emd_8511.map
