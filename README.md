# MoMDeni

Method-of-moments ab initio modeling for cryo-EM from denoised particle images. `kam` denoises the particles, keeps the least crowded
ones and fits a low-resolution map to their first three Fourier-Bessel moments (M1, M2, M3), for example to initialize a refinement.

## Installation

```bash
git clone --recursive https://github.com/wangyuguan/MoMDeni.git
cd MoMDeni
conda create --name momdeni python=3.9 pip
conda activate momdeni
pip install -e .            # CPU
pip install -e ".[gpu]"     # CUDA 12 GPUs (CuPy, cuFINUFFT); PyTorch must be a CUDA build
```

We use the submodules `fle_2d` and `fast-cryoEM-PCA` to denoise images, so clone with `--recursive` and install in editable mode.

## Example

`examples/EMPIAR-10081_denoised_32px.mrcs` holds 10,000 denoised particle images of EMPIAR-10081 (HCN1, C4 symmetry), 32 x 32 pixels
of 10.4 Å:

```bash
kam reconstruct examples/EMPIAR-10081_denoised_32px.mrcs --outdir example_out --symmetry C4
```

The map is `example_out/M3/recon.mrc`. This takes about 3 minutes on an A100 GPU, 16 minutes on 16 CPU cores (`--device cpu`).

## Usage

```bash
kam reconstruct denoised.mrcs --outdir out --symmetry C1
kam pipeline particles.star --datadir /path/to/stacks --outdir results --dataset-label EMPIAR-10081 --symmetry C4
```

`kam reconstruct` takes an MRC stack of square, CTF-corrected (denoised) images. It fits M1 and M2 from 10 random starts, keeps the M2
solution whose model best explains M3 and fits M3 from it; the map is `out/M3/recon.mrc`.

`kam pipeline` takes a RELION particle STAR file with CTF parameters, pick coordinates and micrograph names. It denoises the particles
at 64 x 64 pixels (FastPCA, covariance Wiener filtering), keeps the 10,000 least crowded ones (`--n-select`), denoises them again and
fits their moments as `kam reconstruct` does. The map is `results/<EMPIAR number>_<job id>/kam_pipeline/MoM/M3/recon.mrc`.

Main options (`kam <command> --help` lists all):

- `--symmetry`: point group of the molecule (C1, Cn, Dn, T, O or I).
- `--vol-degree` (7) and `--viewing-degree` (2): degrees of the volume and of the density of viewing directions.
- `--shift-step-px` (0.5) and `--shift-max-px` (side/8): rings of the fitted shift distribution, in pixels of the moment images.
- `--flip-sign` (default) for white particles (RELION and cryoSPARC stacks), `--no-flip-sign` for dark ones.
- `--datadir`, `--strip-prefix` and `--native-pixel-size`: where the image paths of the STAR file point, and its pixel size.
- `--device` (gpu or cpu) and `--seed`.

## Acknowledgements

Our implementation depends on [fle_2d](https://github.com/nmarshallf/fle_2d), [fast-cryoEM-PCA](https://github.com/yunpeng-shi/fast-cryoEM-PCA) and
[ASPIRE](https://github.com/ComputationalCryoEM/ASPIRE-Python).
