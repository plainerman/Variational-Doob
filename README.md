# Doob’s Lagrangian: A Sample-Efficient Variational Approach to Transition Path Sampling
In this work, we propose a novel variational approach to transition path sampling (TPS) based on the Doob’s h-transform. Our method can be used to sample transition paths between two meta-stable states of molecular systems.

![Visualization of alanine dipeptide transitioning between two meta-stable states](visualizations/aldp.gif)

## Setup

You can use the `environment.yml` file to setup this project. However, it only works on CPU.

```bash
conda env create -f environment.yml
```

## Getting started

The best way to get started is to look at the jupyter notebooks which contain code for the Müller-Brown potential.
There is one for the [first order Langevin dynamics](notebooks/tps_gaussian.ipynb) and one for the [second order Langevin dynamics](notebooks/tps_gaussian_2nd.ipynb).

## Running the baselines

To run the baselines (i.e., TPS with shooting) you can run

```bash
python tps_baseline_mueller.py
python eval/evaluate_mueller.py
```

and 

```bash
python tps_baseline.py
python eval/evaluate_tps.py
```

respectively. In both cases, you might need to change the paths that you want to evaluate.

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
