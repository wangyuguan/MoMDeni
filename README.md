# MoMDeni

This repository accompanies the paper

> Yunpeng Shi, Amit Singer, Yuguan Wang and Andy Zhang, *Ab-initio cryo-EM reconstruction from moments of denoised images*,
> [arXiv:2611.25525](https://arxiv.org/abs/2611.25525), 2026.

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
MoM reconstruct examples/EMPIAR-10081_denoised_32px.mrcs --outdir example_out --symmetry C4
```

The map is `example_out/M3/recon.mrc`. This takes about 3 minutes on an A100 GPU, 16 minutes on 16 CPU cores (`--device cpu`).

## Moments of the paper

`examples/moments/` holds the moments M1, M2 and M3 of the six datasets of the paper, formed from the denoised 64 x 64 images of
the selected particles. `MoM fit` runs the moment matching on them with the settings of the paper, e.g.

```bash
MoM fit examples/moments/EMPIAR-10081.npz --outdir fit_10081 --symmetry C4
```

| Moments | Symmetry | Images | Pixel size of the map (Å) |
|---|---|---|---|
| `EMPIAR-10017.npz` | D2 | 9,679 | 4.43 |
| `EMPIAR-10028.npz` | C1 | 10,000 | 7.54 |
| `EMPIAR-10073.npz` | C1 | 10,000 | 8.49 |
| `EMPIAR-10081.npz` | C4 | 10,000 | 5.20 |
| `EMPIAR-11005.npz` | C1 | 10,000 | 6.53 |
| `EMPIAR-11618.npz` | C1 | 10,000 | 3.81 |

The map is `fit_10081/M3/recon.mrc`. Each fit takes 3 to 7 minutes on an H100 GPU.

## Usage

```bash
MoM reconstruct denoised.mrcs --outdir out --symmetry C1
MoM fit moments.npz --outdir out --symmetry C1
MoM pipeline particles.star --datadir /path/to/stacks --outdir results --dataset-label EMPIAR-10081 --symmetry C4
```

`MoM reconstruct` takes an MRC stack of square, CTF-corrected (denoised) images. It fits M1 and M2 from 10 random starts, keeps the M2
solution whose model best explains M3 and fits M3 from it; the map is `out/M3/recon.mrc`. `MoM fit` does the same from precomputed
moments (`out/prepared/moments.npz` of an earlier run).

`MoM pipeline` takes a RELION particle STAR file with CTF parameters, pick coordinates and micrograph names. It denoises the particles
at 64 x 64 pixels (FastPCA, covariance Wiener filtering), keeps the 10,000 least crowded ones (`--n-select`), denoises them again and
fits their moments as `MoM reconstruct` does. The map will be `results/<EMPIAR number>_<job id>/kam_pipeline/MoM/M3/recon.mrc`.

Main options (`MoM <command> --help` lists all):

- `--symmetry`: point group of the molecule (C1, Cn, Dn, T, O or I).
- `--vol-degree` (7) and `--viewing-degree` (2): degrees of the volume and of the density of viewing directions.
- `--shift-step-px` (0.5) and `--shift-max-px` (side/8): rings of the fitted shift distribution, in pixels of the moment images.
- `--flip-sign` (default) for white particles (RELION and cryoSPARC stacks), `--no-flip-sign` for dark ones.
- `--datadir`, `--strip-prefix` and `--native-pixel-size`: where the image paths of the STAR file point, and its pixel size.
- `--device` (gpu or cpu) and `--seed`.

## Acknowledgements

Our implementation depends on [fle_2d](https://github.com/nmarshallf/fle_2d), [fast-cryoEM-PCA](https://github.com/yunpeng-shi/fast-cryoEM-PCA) and
[ASPIRE](https://github.com/ComputationalCryoEM/ASPIRE-Python).
