# Doob’s Lagrangian: A Sample-Efficient Variational Approach to Transition Path Sampling
In this work, we propose a novel variational approach to transition path sampling (TPS) based on the Doob’s h-transform. Our method can be used to sample transition paths between two meta-stable states of molecular systems.

![Visualization of alanine dipeptide transitioning between two meta-stable states](visualizations/aldp.gif)

## Setup

You can use the `environment.yml` file to setup this project. However, it only works on CPU.
```bash
conda env create -f environment.yml
```

We also provide a requirements.txt, however, it is not guaranteed to work.

## Getting started

The best way to get started is to look at [the jupyter notebook](notebooks/tps_gaussian.ipynb) which contains the necessary code for 2D potentials in one place.

## Running the baselines

To run the baselines (i.e., TPS with shooting) you can run

```bash
python tps_baseline_mueller.py
# For this to work, you need to specify which baselines to run by changing the all_paths variable in the script
python eval/evaluate_mueller.py
```

for a toy-potential and 

```bash
python tps_baseline.py --mechanism two-way-shooting --num_paths 1000 --states phi-psi
# num_steps compiles multiple MD steps into a single one. This makes sampling faster but increases startup time. Only really worth it for long running simulations
python tps_baseline.py --mechanism two-way-shooting --num_paths 100 --fixed_length 1000 --states phi-psi --num_steps 50
python tps_baseline.py --mechanism two-way-shooting --num_paths 1000 --states rmsd
python eval/evaluate_tps.py
```

for ALDP respectively. In both cases, you might need to change the paths that you want to evaluate.

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
