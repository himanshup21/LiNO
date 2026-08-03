#!/usr/bin/env bash
# Download / prepare datasets for all LiNO benchmarks.
# Edit paths and URLs to match your actual hosting once finalized.
set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"/{darcy,poisson,allen_cahn,navier_stokes,gray_scott}

echo "=== Darcy flow & Allen-Cahn (Tripura & Chakraborty / Li et al. FNO datasets) ==="
# Source: https://github.com/neuraloperator/neuraloperator (FNO datasets)
# Source: https://github.com/Katiana22/wavelet-neural-operator (WNO / Allen-Cahn dataset)
echo "-> See README section 'Datasets' for direct links; download manually or add wget/curl calls here."

echo "=== Poisson equation ==="
echo "-> Synthetically generated via lino/data/preprocessing.py; run:"
echo "   python -m lino.data.preprocessing --benchmark poisson --n_train 800 --n_test 200"

echo "=== Compressible Navier-Stokes (PDEBench) ==="
# Source: https://github.com/pdebench/PDEBench
echo "-> Requires PDEBench download utility. See:"
echo "   https://github.com/pdebench/PDEBench#download-dataset"

echo "=== Gray-Scott (The Well) ==="
# Source: https://github.com/PolymathicAI/the_well
echo "-> Requires 'the_well' package. Install via: pip install the-well"
echo "   Then: python -m the_well.download --dataset gray_scott_reaction_diffusion"

echo "Done. Populate $DATA_DIR/<benchmark>/ before running training scripts."
