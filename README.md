<div align="center">

# tensorframes

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.*-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

</div>

## Description

The `tensorframes` package implements the message passing class described in the paper https://arxiv.org/abs/2405.15389v1. This class generalizes the typical message passing algorithm by transforming features from one node's local frame to another node's frame. This transformation results in an $O(N)$ invariant layer, which can be used to construct fully equivariant architectures:

$$
f_i^{(k)}=\psi^{(k)}\bigg( f_i^{(k-1)}, \bigoplus_{j\in\mathcal{N}}\phi^{(k)}\left(f_i^{(k-1)},\rho(g_i g_j^{-1})f_j^{(k-1)}, \rho_e(g_i)e_{ji}, R_i(\mathbf x_i - \mathbf x_j)\right) \bigg)
$$

The `TFMessagePassing` class is introduced to efficiently implement these layers, abstracting the transformation behavior of the parameters. For predicting local frames, the `LearnedLFrames` module is available, which calculates the local frame based on a local neighborhood. Additionally, we provide input and output layers to build fully end-to-end equivariant models, adhering to the guidelines outlined in the referenced paper.

### Create your own module

The whole transformations are abstracted away by the `TFMessagePassing` class, where every parameter is transformed into the right frame. A simple GCNConv-like module could look the following:

```python
from tensorframes.nn.tfmessage_passing import TFMessagePassing
from tensorframes.reps.tensorreps import TensorReps

class GCNConv(TFMessagePassing):
    def __init__(self, in_reps: TensorReps, out_reps: TensorReps):
        super().__init__(
            params_dict={
                "x": {"type": "local", "rep": in_reps}
            }
        )
        self.linear = torch.nn.Linear(in_reps.dim, out_reps.dim)

    def forward(self, edge_index, x, lframes):
        return self.propagate(edge_index, x=x, lframes=lframes)

    def message(self, x_j):
        return self.linear(x_j)

module = GCNConv(TensorReps("16x0n+8x1n"), TensorReps("4x0n+1x1n"))
```

Here the feature `x_j` is automatically transformed into the local frame of node i. The transformation behavior of the parameters which are parsed in the propagate function can be determined by the `params_dict`.

## Installation

#### Clone Project

```bash
git clone https://github.com/sciai-lab/tensorframes
cd tensorframes
```

#### Install using Conda/Mamba/Micromamba

For mamba or micromamba replace `conda` with `mamba` or `micromamba` below. (Micromamba is recommended)

```bash
# create conda environment and install dependencies
conda env create -f environment.yaml -n tensorframes

# activate conda environment
conda activate tensorframes

# install as an editable package
pip install -e .
```

## Developer Info

#### Pre-commit

Before starting to commit, run

```bash
pre-commit install
```

After installing, pre-commit will check the code formatting and much more when you try to commit.
Exact settings can be found in .pre-commit-config.yaml.
If `pre-commit` fails during committing, check for changed files, stage them again and commit again.
If it still fails, read what is failing and manually fix it. If you want to commit anyway run:

```bash
git commit --no-verify -m "message"
```

#### Testing

To go through the tests of the package just run:

```bash
pytest
```
