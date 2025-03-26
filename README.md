<h1 align="center">Doob’s Lagrangian: A Sample-Efficient Variational Approach to Transition Path Sampling</h1>
<p align="center">
<a href="https://arxiv.org/abs/2410.07974"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"/></a>
<!-- <a href="https://github.com/plainerman/variational-doob"><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/></a> -->
<a href="https://colab.research.google.com/drive/1FcmEbec06cH4yk0t8vOIt8r1Gm-VjQZ0?usp=sharing"><img src="https://img.shields.io/badge/Colab-e37e3d.svg?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Jupyter"/></a>
<a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/library-JAX-5f0964?style=for-the-badge" alt="Jax"/></a>
</p>
<p align="center">
A novel variational approach to transition path sampling (TPS) based on the Doob’s h-transform. Our method can be used to sample transition paths between two meta-stable states of molecular systems.
</p>
<p align="center">
<img src="visualizations/aldp.gif" alt="Visualization of alanine dipeptide transitioning between two meta-stable states"/>
</p>
<p align="center">
<i>A transition path of alanine dipeptide sampled using our method.</i>
</p>

<p align="center">
<img src="visualizations/training-optimized.gif" alt=""/>
</p>
<p align="center">
<i>Visualization of the optimization process using our algorithm for 2D potential.</i>
</p>

<p align="center">
<img src="visualizations/simulation-optimized.gif" alt=""/>
</p>
<p align="center">
<i>Running the deterministic and stochastic simulations using our algorithm for 2D potential.</i>
</p>

# FAQ
## I am getting NaN values when running experiments on alanine dipeptide!
This is an issue on certain devices, and, so far, we haven't figured out the underlying reason. However, we have found out that:

1. Changing your floats to 64-bit precision prevents this problem from happening (at least on our machines), albeit at ~2x slower performance. To change to float64, simply search for all instances of `jnp.float32` (as can be seen [here](https://github.com/search?q=repo%3Aplainerman%2FVariational-Doob%20jnp.float32&type=code)) and change it to `jnp.float64`.

2. First-order systems usually do not exhibit this behavior. So you can also change your `ode` in the config (e.g., [here](https://github.com/plainerman/Variational-Doob/blob/b3836998080569af5deaaa5bd1ef6ad0993e0bd9/configs/aldp_diagonal_single_gaussian.yaml#L7)) to `first_order` and see if this resolves the issue.  In our tests, first-order ODE was sufficient for most setups. 

# Getting started

The best way to understand our method is to look at [the google colab notebook](https://colab.research.google.com/drive/1FcmEbec06cH4yk0t8vOIt8r1Gm-VjQZ0?usp=sharing) which contains the necessary code for 2D potentials in one place. 
However, this notebook is very limited in scope and only contains the most basic examples. In the following, we will show the interfaces to run more complex examples. You can also look at the setups in the `configs/` folder.


# Setup

You can use the `environment.yml` file to setup this project. However, it only works on CPU.
```bash
conda env create -f environment.yml
```

We also provide a requirements.txt, and a pyproject.toml. So if you are using [pixi](https://github.com/prefix-dev/pixi) you can instead run

```bash
pixi install --frozen
```

to install the dependencies and setup a virtual environment. Either activate the environment with `pixi shell` or use the provided `pixi run` command to run the scripts.

# Running the code

## Baselines
You can either use the TPS shooting baselines [provided by us](https://github.com/plainerman/variational-doob/releases/tag/camera-ready), or re-create them by running

```bash
python tps_baseline_mueller.py
PYTHONPATH='.' python eval/evaluate_mueller.py
```

to generate and evaluate transitions for the Müller-Brown toy-potential or use

```bash
python tps_baseline.py --mechanism two-way-shooting --num_paths 1000 --states phi-psi
# num_steps compiles multiple MD steps into a single one. This makes sampling faster but increases startup time. Only really worth it for long running simulations
python tps_baseline.py --mechanism two-way-shooting --num_paths 100 --fixed_length 1000 --states phi-psi --num_steps 50
python tps_baseline.py --mechanism two-way-shooting --num_paths 1000 --states rmsd
PYTHONPATH='.' python eval/evaluate_tps.py
```

for ALDP respectively. 

**Note:** In both cases, you might want to change the paths that you want to generate and evaluate in the baseline or evaluation scripts.

## Our Method
To sample trajectories with our method, we provide ready to go config files in `configs/`. You can run them with

```bash
python main.py --config configs/toy/mueller_single_gaussian.yaml
python main.py --config configs/toy/dual_channel_single_gaussian.yaml
python main.py --config configs/toy/dual_channel_two_gaussians.yaml
```

for the toy examples and

```bash
python main.py --config configs/aldp_diagonal_single_gaussian.yaml
```

for real molecular systems.

# Citation
If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{du2024doob,
  author = {Du, Yuanqi and Plainer, Michael and Brekelmans, Rob and Duan, Chenru and No{\'e}, Frank and Gomes, Carla P. and Aspuru-Guzik, Al{\'a}n and Neklyudov, Kirill},
  title = {Doob’s Lagrangian: A Sample-Efficient Variational Approach to Transition Path Sampling},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {Globerson, A. and Mackey, L. and Belgrave, D. and Fan, A. and Paquet, U. and Tomczak, J. and Zhang, C.},
  pages = {65791--65822},
  publisher = {Curran Associates, Inc.},
  volume = {37},
  year = {2024}
}
```
