#!/usr/bin/env bash
#SBATCH --job-name=kam10028
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
RUN_ROOT="${RESULTS}/10028_${SLURM_JOB_ID}"
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
kam pipeline /home/yw3384/datasets/EMPIAR-10028/shiny_2sets_fixed.star \
  --dataset-label EMPIAR-10028 \
  --datadir /home/yw3384/datasets/EMPIAR-10028 \
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

# Native-grid RELION inputs and initial low-pass.
kam relion_prepare \
  --pipeline-dir "$RUN_ROOT/kam_pipeline" \
  --particle-diameter 300 \
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
  --molecular-weight 3.5e6

# Optional reference comparison.
kam align_reference \
  --relion-dir "$RUN_ROOT/relion_refine" \
  --reference-map /home/yw3384/softwares/Kam-cryoEM/reference_map/emd_2660.map
