# MMDI

Method-of-moments (MoM) ab initio modeling for cryo-EM from denoised particle images. The command-line tool `kam` denoises the particles,
selects the least crowded ones, and fits a low-resolution volume to their first three Fourier-Bessel moments (M1, M2, M3); the volume is an ab initio
model, for example to initialize a refinement.

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
`git submodule update --init`) and install in editable mode.

Testing install

`examples/EMPIAR-10081_denoised_32px.mrcs` holds 10,000 denoised particle images of EMPIAR-10081 (HCN1 channel, C4 symmetry),
the images `kam pipeline` selected and denoised, Fourier-cropped to 32 x 32 pixels (10.4 A per pixel). Reconstruct a map from them with

```bash
kam reconstruct examples/EMPIAR-10081_denoised_32px.mrcs --outdir example_out --symmetry C4
```

This takes about 3 minutes on one A100 GPU (about 16 minutes on 16 CPU cores with `--device cpu`). The map is `example_out/M3/recon.mrc` (32^3 voxels at
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

# Acknowledgements

Denoising uses [fle_2d](https://github.com/nmarshallf/fle_2d) (fast Fourier-Bessel expansion), [fast-cryoEM-PCA](https://github.com/yunpeng-shi/fast-cryoEM-PCA)
and [ASPIRE](https://github.com/ComputationalCryoEM/ASPIRE-Python).
