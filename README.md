<h1 align="center">Doob’s Lagrangian: A Sample-Efficient Variational Approach to Transition Path Sampling</h1>
<p align="center">
<a href="https://github.com/plainerman/variational-doob"><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Built with Python"/></a>
<a href="https://github.com/plainerman/variational-doob/blob/main/notebooks/tps_gaussian.ipynb"><img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/></a>
<a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/library-JAX-5f0964?style=for-the-badge" alt="Jax"/></a>
</p>
<p align="center">
A novel variational approach to transition path sampling (TPS) based on the Doob’s h-transform. Our method can be used to sample transition paths between two meta-stable states of molecular systems.
</p>
<p align="center">
<img src="visualizations/aldp.gif" alt="Visualization of alanine dipeptide transitioning between two meta-stable states"/>
</p>

## Setup

You can use the `environment.yml` file to setup this project. However, it only works on CPU.
```bash
conda env create -f environment.yml
```

We also provide a requirements.txt, and a pyproject.toml. So if you are using [uv](https://github.com/astral-sh/uv) you can instead run

```bash
uv sync
```

to install the dependencies and setup a virtual environment. Either activate the environment or use the provided `uv run` command to run the scripts.

## Getting started

The best way to get started is to look at [the jupyter notebook](notebooks/tps_gaussian.ipynb) which contains the necessary code for 2D potentials in one place.

## Running the baselines
You can either use the TPS shooting baselines [provided by us](https://github.com/plainerman/variational-doob/releases/tag/camera-ready), or re-create them by running

```bash
python tps_baseline_mueller.py
# For this to work, you need to specify which baselines to run by changing the all_paths variable in the script
python eval/evaluate_mueller.py
```

to generate and evaluate transitions for the Müller-Brown toy-potential or use

```bash
python tps_baseline.py --mechanism two-way-shooting --num_paths 1000 --states phi-psi
# num_steps compiles multiple MD steps into a single one. This makes sampling faster but increases startup time. Only really worth it for long running simulations
python tps_baseline.py --mechanism two-way-shooting --num_paths 100 --fixed_length 1000 --states phi-psi --num_steps 50
python tps_baseline.py --mechanism two-way-shooting --num_paths 1000 --states rmsd
python eval/evaluate_tps.py
```

for ALDP respectively. 

**Note:** In both cases, you might need to change the paths that you want to evaluate in ``evaluate_mueller.py` or `evaluate_tps.py`.

## Run our method
To sample trajectories with our method, we provide ready to go config files in `configs/`. You can run them with

```bash
python main.py --config configs/toy/mueller_single_gaussian.yaml
python main.py --config configs/toy/dual_channel_single_gaussian.yaml
python main.py --config configs/toy/dual_channel_two_gaussian.yaml
```

for the toy examples and

```bash
python main.py --config configs/aldp_diagonal_single_gaussian.yaml
```

for real molecular systems.
