# LiNO: Lifting-based Multiresolution Neural Operator

[![arXiv](https://img.shields.io/badge/arXiv-2607.02715-b31b1b.svg)](https://arxiv.org/abs/2607.02715)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LiNO**, a lifting-based neural operator framework that integrates the second-generation wavelet into neural operator learning, providing a fully invertible and learnable multiresolution decomposition tailored to PDE training data. It offers a scale-aware operator evolution mechanism in the lifted multiresolution space that separately propagates coarse and directional detail coefficients while preserving localized physical interactions.


---


<p align="center">
  <img src="docs/figures/architecture.png" alt="LiNO architecture" width="800"/>
</p>

*Figure: the input field is encoded (E), decomposed via multi-level learnable lifting (L),
evolved in the lifted space by the operator K_Φ, reconstructed by the inverse lifting (L⁻¹), and
decoded (D) back to the solution field. See Fig. 2.1 in the paper for full detail.*

## What LiNO can do

LiNO has been evaluated on benchmark PDEs spanning elliptic, reaction–diffusion, and
compressible fluid dynamics regimes:

| Benchmark | Physical regime | Highlights |
|---|---|---|
| Darcy flow | Elliptic, heterogeneous media | Steady-state pressure field prediction |
| Poisson equation | Elliptic, localized sources | Sharp, localized potential structures |
| Allen–Cahn | Reaction–diffusion, phase separation | Sharp moving interfaces |
| Compressible Navier–Stokes (viscous & inviscid) | Transport-dominated, chaotic | Long-horizon autoregressive rollout |
| Gray–Scott reaction–diffusion | Nonlinear pattern formation | Spiral structure formation, long-range dependencies |

Across these benchmarks, LiNO achieves competitive or state-of-the-art relative ℓ₂-error against
FNO, WNO, UNO, CNO, and LNO baselines, with substantially fewer trainable parameters than
comparable multiscale architectures. Full quantitative results are in Tables 4.1–4.2 and
Appendix B of the paper.

**Known limitations** (see Section 5 of the paper): the current implementation requires
**dyadic (power-of-two) grid resolutions** and is restricted to **structured Cartesian grids**.
Extending LiNO to irregular meshes and non-dyadic resolutions is ongoing work.

## Installation

```bash
git clone https://github.com/<org-or-username>/LiNO.git
cd LiNO
conda env create -f environment.yml
conda activate lino
pip install -e .
```

Requires Python ≥3.10 and a CUDA-capable GPU (paper results were produced on an NVIDIA RTX 4500,
32 GB, CUDA 12.3).

## Quickstart

```python
import torch
from lino.models.lino import LiNO
from lino.data.datasets import load_darcy

# Load a pretrained checkpoint
model = LiNO.from_pretrained("checkpoints/darcy_flow.pt")
model.eval()

# Run inference on a test sample
a, u_true = load_darcy(split="test")[0]
with torch.no_grad():
    u_pred = model(a.unsqueeze(0))

print("Relative L2 error:", (u_pred - u_true).norm() / u_true.norm())
```

See `notebooks/quickstart_demo.ipynb` for a full walkthrough with visualization.

## Reproducing the paper's results

Each benchmark has a dedicated config matching the hyperparameters in Table A.3 of the paper.

```bash
# Train LiNO on a given benchmark
python -m lino.train --config configs/darcy.yaml

# Evaluate a trained checkpoint and reproduce Table 4.1 entries
python -m lino.evaluate --config configs/darcy.yaml --checkpoint checkpoints/darcy_flow.pt

# Reproduce the full Table 4.1 (all benchmarks, all baselines)
bash scripts/reproduce_table_4_1.sh
```

| Paper artifact | Reproduction script |
|---|---|
| Table 4.1 (relative ℓ2-error) | `scripts/reproduce_table_4_1.sh` |
| Table 4.2 (training time) | `scripts/reproduce_table_4_2.sh` |
| Table B.1 / B.2 (memory, parameter count) | `scripts/reproduce_appendix_b.sh` |
| Figure 4.2 / 4.3 (qualitative comparisons) | `notebooks/reproduce_figures.ipynb` |

## Datasets

| Benchmark | Source | Resolution |
|---|---|---|
| Allen–Cahn | Tripura & Chakraborty (WNO paper) dataset | 128×128 |
| Darcy flow | Li et al. (FNO paper) dataset | 128×128 |
| Poisson equation | Synthetically generated (see `lino/data/preprocessing.py`) | 128×128 |
| Compressible Navier–Stokes (viscous/inviscid) | PDEBench | 128×128 (downsampled from 512×512) |
| Gray–Scott | The Well (Ohana et al.) | 128×128 |

Download instructions and preprocessing scripts are in `scripts/download_data.sh`.

## Repository structure

```
LiNO/
├── lino/               # Core library: models, data loaders, training/eval loops
├── configs/            # Per-benchmark hyperparameter configs (matches Table A.3)
├── scripts/            # Data download and result-reproduction scripts
├── notebooks/          # Quickstart and figure-reproduction notebooks
├── tests/              # Unit tests, including numerical invertibility check
├── checkpoints/        # Pretrained model weights (or see Zenodo archive)
└── docs/               # Architecture notes and extended documentation
```

## Citation

If you use LiNO in your research, please cite:

```bibtex
@article{pandey2026lino,
  title   = {LiNO: Lifting based multiresolution neural operator},
  author  = {Pandey, Himanshu and Patel, Subham and Behera, Ratikanta},
  journal = {arXiv preprint arXiv:2607.02715},
  year    = {2026}
}
```

A `CITATION.cff` file is also provided for GitHub's native citation widget.

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

Subham Patel acknowledges the Axis Bank Centre for Mathematics and Computing, Indian Institute
of Science, Bangalore, for financial support carried out at the IISc Mathematics Initiative,
Department of Mathematics.
