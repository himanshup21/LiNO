# LiNO: Lifting-based Multiresolution Neural Operator

[![arXiv](https://img.shields.io/badge/arXiv-2607.02715-b31b1b.svg)](https://arxiv.org/abs/2607.02715)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LiNO, a lifting-based neural operator framework that integrates the second-generation wavelet into neural operator learning, providing a fully invertible and learnable multiresolution decomposition tailored to PDE training data. It offers a scale-aware operator evolution mechanism in the lifted multiresolution space that separately propagates coarse and directional detail coefficients while preserving localized physical interactions.


---

<p align="center">
  <img src="docs/LiNO-architech.png" alt="LiNO architecture" width="600"/>
</p>

*Architechure: The input field $`a(x)`$ is encoded into a latent representation using $`\mathcal{E}`$. The learnable predict ($`\mathcal{P}_\omega`$) and update ($`\mathcal{U}_\phi`$) operators, augmented with spatial coordinates $`g(x,y)`$, perform adaptive multiscale decomposition and reconstruction. Operator learning is carried out in the lifted space using the kernel operator $`\mathcal{K}_{\Phi}`$, followed by a multi-level inverse lifting transform and a decoder $`\mathcal{D}`$ to obtain the solution field $`u({x})`$. The upper and lower insets illustrate the forward and inverse lifting procedures, respectively.*



Requires Python ≥3.10 and a CUDA-capable GPU (paper results were produced on an NVIDIA RTX 4500,
32 GB, CUDA 12.3).

## Quickstart

```python
from lino_darcy import*

# update the path to the saved model
checkpoint = torch.load(best_ckpt_path)

params = checkpoint['model_state']
train_loss = checkpoint['loss']

model.load_state_dict(params)

# model evaluation and visualization over 3 randomly sampled test inputs
model_eval(epoch=epochs, loss=train_loss, t0=time.time(), inference=True)
visualization()
```


## Reproducing the paper's results

Each benchmark has a dedicated config matching the hyperparameters in Table A.3 of the paper.


## Datasets

| Benchmark | Source | Resolution |
|---|---|---|
| Allen–Cahn | Tripura & Chakraborty (WNO paper) dataset | 128×128 |
| Darcy flow | Li et al. (FNO paper) dataset | 128×128 |
| Poisson equation | Synthetically generated (see `lino/data/preprocessing.py`) | 128×128 |
| Compressible Navier–Stokes (viscous/inviscid) | PDEBench | 128×128 (downsampled from 512×512) |
| Gray–Scott | The Well (Ohana et al.) | 128×128 |

Download instructions and preprocessing scripts are in `scripts/download_data.sh`.



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


## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

Subham Patel acknowledges the Axis Bank Centre for Mathematics and Computing, Indian Institute
of Science, Bangalore, for financial support carried out at the IISc Mathematics Initiative,
Department of Mathematics.
