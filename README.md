# MMDI

Method-of-moments (MoM) ab initio modeling for cryo-EM from denoised particle images. The command-line tool `kam` denoises the particles,
selects the least crowded ones, fits a low-resolution volume to their first three Fourier-Bessel moments (M1, M2, M3), and prepares a RELION
refinement initialized from that volume.

Installing in conda environment

```bash
git clone --recursive https://github.com/wangyuguan/MMDI.git
cd MMDI
conda create --name mmdi python=3.9 pip
conda activate mmdi
pip install -e .            # CPU
pip install -e ".[gpu]"     # CUDA 12 GPUs (CuPy, cuFINUFFT); PyTorch must be a CUDA build
```

The denoising step uses the submodules `fle_2d` and `fast-cryoEM-PCA` from the checkout, so clone with `--recursive` (or run
`git submodule update --init`) and install in editable mode. RELION refinement needs RELION 5.

Testing install

`examples/EMPIAR-10081_denoised_32px.mrcs` holds 10,000 denoised particle images of EMPIAR-10081 (HCN1 channel, C4 symmetry),
the images `kam pipeline` selected and denoised, Fourier-cropped to 32 x 32 pixels (10.4 A per pixel). Reconstruct a map from them with

```bash
kam reconstruct examples/EMPIAR-10081_denoised_32px.mrcs --outdir example_out --symmetry C4
```

This takes about 3 minutes on one A100 GPU (`--device cpu` also works, more slowly). The map is `example_out/M3/recon.mrc` (32^3 voxels at
10.4 A).


## Usage

`kam` has one subcommand per step. Every subcommand prints all of its options with `kam <subcommand> --help`.

### MoM map from a stack of denoised images

```bash
kam reconstruct denoised.mrcs --outdir out --symmetry C1
```

`denoised.mrcs` is an MRC stack of square, CTF-corrected (denoised) particle images; the pixel size is read from its header. The command
forms M1, M2 and M3 of the stack (`out/prepared/moments.npz`) and fits them in turn:

1. M1 and M2 from 10 random starts, start i of M2 beginning at the M1 fit i (`out/M1`, `out/M2`, at most 10000 L-BFGS iterations each);
2. the M2 solution whose model M3 (with its least-squares amplitude) is closest to the data M3, before any M3 fitting, is selected
   (`out/M2_selected`; `selection.json` lists every start's M3 residual);
3. M3 from the selected solution (`out/M3`, at most 500 L-BFGS iterations).

The map is `out/M3/recon.mrc`.

- `--symmetry`: point group of the molecule (C1, Cn, Dn, T, O or I); the volume is restricted to it.
- `--vol-degree` (default 7): maximum angular degree of the spherical Fourier-Bessel expansion of the volume.
- `--viewing-degree` (default 2): degree of the spherical-harmonic density of viewing directions.
- `--moment-res` (default: the side of the stack): side of the images the moments are formed on.
- `--shift-step-px` (default 0.5) and `--shift-max-px` (default side/8): the particle shifts are modeled by an isotropic distribution on
  rings with this spacing and maximum radius (pixels of the moment images), fitted together with the volume and the viewing density.
- `--device` (cpu or gpu, default gpu) and `--seed` (default 0).

### MoM map from a RELION particle STAR file

```bash
kam pipeline particles.star --datadir /path/to/stacks --outdir results --dataset-label EMPIAR-10081 --symmetry C4
```

`particles.star` is a RELION particle STAR file with CTF parameters, pick coordinates and micrograph names; `--datadir` is the folder the
image paths in it are relative to (default: the folder of the STAR file). The pipeline

1. Fourier-crops the particles to 64 x 64 pixels and denoises them with FastPCA / covariance Wiener filtering (CWF);
2. selects the `--n-select` (default 10000) least crowded particles: picks closer than 280/380 of the box to another pick are discarded
   (`--neighbor-distance-fraction`), and the rest are ranked by a crowding score, the signal near the edge of the denoised image relative
   to its center; if fewer particles pass than requested, all of them are used;
3. denoises the selected particles again, on their own;
4. forms M1, M2 and M3 on 64 x 64 images and fits them as `kam reconstruct` does (10 starts of M1 and M2, selection of one M2 solution
   by its M3 residual, then M3; same options and defaults).

The output is `results/<EMPIAR number or label>_<Slurm job id>/kam_pipeline`, with the map in `kam_pipeline/MoM/M3/recon.mrc`
(outside Slurm, a local run identifier replaces the job id). Settings go to `hyperparameters.json` and progress to `run.log`.
`--dry-run` checks the inputs and prints the stages without running them.

Other options:

- `--flip-sign` (default) / `--no-flip-sign`: the pipeline expects dark particles on a light background; stacks extracted by RELION or
  cryoSPARC (white particles) need the default `--flip-sign`.
- `--strip-prefix` and `--native-pixel-size`: strip a leading folder from the image paths of the STAR file, or override its pixel size.
- `--denoise-res`, `--moment-res`: sides of the denoised and the moment images (default 64 for both).

### RELION refinement from the MoM map

```bash
kam relion_prepare --pipeline-dir results/10081_<job>/kam_pipeline
bash results/10081_<job>/relion_prepare/run_refinement.sh
```

`kam relion_prepare` resamples the MoM map to the box and pixel size of the raw particles and writes `relion_prepare/` next to
`kam_pipeline/`: the initial model, a particle STAR file without poses or origins, and `run_refinement.sh`, a RELION 5 3D auto-refine of all
raw particles (initial low-pass 60 A, angular sampling 7.5 degrees with local searches from 1.8 degrees, translations up to 1/8 of the box
in steps of 1/40 of the box). The mask diameter is computed from the first moment: 1.25 times the diameter that holds 99% of the
positive mass of the mean particle image (the Hankel transform of M1). `--symmetry` defaults to the symmetry of the MoM map; `--particles` and `--datadir` refine another STAR file (for example the full-size particles when the pipeline read a downsampled
copy). `relion_prepare` must run inside a Slurm job, normally the job of the pipeline: the refinement script copies that job's
`#SBATCH` header (the refinement needs 2 GPUs and 24 CPUs) and loads RELION 5 through the environment modules of Princeton's Della
cluster, so edit its `module` lines on other systems. Started outside a job, `run_refinement.sh` submits itself with `sbatch`.

After the refinement,

```bash
kam postprocess --relion-dir results/10081_<job>/relion_refine --molecular-weight 2.99e5
kam align_reference --relion-dir results/10081_<job>/relion_refine --reference-map emd_8511.map
```

`kam postprocess` makes a solvent mask from the refined map (its volume set by `--molecular-weight`, in Daltons) and runs
`relion_postprocess` for the masked FSC resolution. `kam align_reference` aligns the MoM map and the refined map to a reference map and
plots their FSC curves; the reference is used only for this comparison.

A complete Slurm job for one dataset runs the commands above in order:

```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=12:00:00
set -euo pipefail
RESULTS=results
RUN="$RESULTS/10081_${SLURM_JOB_ID}"
kam pipeline particles.star --datadir stacks --outdir "$RESULTS" --dataset-label EMPIAR-10081 --symmetry C4 --device gpu --workers 16
kam relion_prepare --pipeline-dir "$RUN/kam_pipeline"
bash "$RUN/relion_prepare/run_refinement.sh"
kam postprocess --relion-dir "$RUN/relion_refine" --molecular-weight 2.99e5   # with relion_postprocess on the PATH
kam align_reference --relion-dir "$RUN/relion_refine" --reference-map emd_8511.map
```

# Acknowledgements

Denoising uses [fle_2d](https://github.com/nmarshallf/fle_2d) (fast Fourier-Bessel expansion), [fast-cryoEM-PCA](https://github.com/yunpeng-shi/fast-cryoEM-PCA)
and [ASPIRE](https://github.com/ComputationalCryoEM/ASPIRE-Python).
