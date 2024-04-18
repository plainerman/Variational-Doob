import os
import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import trange

# install openmm (from conda)
import openmm.app as app
import openmm.unit as unit
# install dmff (from source)
from dmff import Hamiltonian, NeighborList
# install mdtraj
import mdtraj as md
# helper function for plotting in 2D
from utils.PlotPathsAlanine_jax import PlotPathsAlanine

class MLPq(nn.Module):
    @nn.compact
    def __call__(self, t):
        t = t/T
        h = nn.Dense(128)(t-0.5)
        h = nn.swish(h)
        h = nn.Dense(128)(h)
        h = nn.swish(h)
        h = nn.Dense(128)(h)
        h = nn.swish(h)
        h = nn.Dense(67)(h)
        mu = (1-t)*A[:BS] + t*B[:BS] + (1-t)*t*h[:,:66]
        sigma = (1-t)*1e-2 + t*1e-2 + (1-t)*t*jnp.exp(h[:,66:])
        return mu, sigma

# working dir
savedir = f"models_variational_gp/alanine"
os.makedirs(savedir, exist_ok=True)

# read initial and target files 
init_pdb = app.PDBFile("./files/AD_c7eq.pdb")
target_pdb = app.PDBFile("./files/AD_c7ax.pdb")

# Hyperparameters
T = 2.0
dt = 1e-2
BS = 512
lr = 1e-3
clip_value = 1e8 # clip force
n_iterations = 200
# Boltzmann constant
temp = 298.15
kbT = 1.380649 * 6.02214076 * 1e-3 * temp
# Construct the mass matrix
mass = [a.element.mass.value_in_unit(unit.dalton) for a in init_pdb.topology.atoms()]
new_mass = []
for mass_ in mass:
    for _ in range(3):
        new_mass.append(mass_)
mass = jnp.array(new_mass)
# Obtain sigma, gamma is by default 1
sigma = jnp.sqrt(2 * kbT / mass)

# Initial and target shape [BS, 66]
A = jnp.array(init_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(1, -1)
A = jnp.tile(A, (BS,1))
B = jnp.array(target_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(1, -1)
B = jnp.tile(B, (BS,1))
print (A.shape, B.shape)

# Initialize the potential energy with amber forcefields
ff = Hamiltonian('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
potentials = ff.createPotential(init_pdb.topology,
                                nonbondedMethod=app.NoCutoff,
                                nonbondedCutoff=1.0 * unit.nanometers,
                                constraints=None,
                                ewaldErrorTolerance=0.0005)
U = potentials.getPotentialFunc()
# Create a box used when calling 
# Calling U by U(x, box, pairs, ff.paramset.parameters), x is [22, 3] and output the energy, if it is batched, use vmap
box = np.array([[50.0,0.0,0.0],[0.0,50.0,0.0],[0.0,0.0,50.0]])
nbList = NeighborList(box, 4.0, potentials.meta["cov_map"])
nbList.allocate(init_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
pairs = nbList.pairs

# Initialization of network and optimizer
q = MLPq()
key = jax.random.PRNGKey(1)
key, *init_key = jax.random.split(key, 3)
params_q = q.init(init_key[0], jnp.ones([BS, 1]))

optimizer_q = optax.adam(learning_rate=lr)
state_q = train_state.TrainState.create(apply_fn=q.apply,
                                        params=params_q,
                                        tx=optimizer_q)

def loss_fn(params_q, key):
    key = jax.random.split(key)
    t = T*jax.random.uniform(key[0], [BS,1])
    eps = jax.random.normal(key[1], [BS,66])
    mu_t = lambda _t: state_q.apply_fn(params_q, _t)[0]
    sigma_t = lambda _t: state_q.apply_fn(params_q, _t)[1]

    def dmudt(_t):
        _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0))
        return _dmudt(_t).squeeze().T

    dUdx_fn = jax.grad(lambda _x: U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters).sum())
    dUdx_fn = jax.vmap(dUdx_fn)

    dsigmadt = jax.grad(lambda _t: sigma_t(_t).sum())
    def v_t(_eps, _t):
        u_t = dmudt(_t) + dsigmadt(_t)*_eps
        _x = mu_t(_t) + sigma_t(_t)*_eps
        # TODO: clip the vector by some predefined values (clip_value)
        out = u_t + jnp.clip(dUdx_fn(_x), -clip_value, clip_value)/mass - 0.5*(sigma**2)*_eps/sigma_t(t)
        # without clip
        # out = u_t + (dUdx_fn(_x)/mass) - 0.5*(sigma**2)*_eps/sigma_t(t)
        return out
    loss = 0.5*((v_t(eps, t)/sigma)**2).sum(1, keepdims=True)
    return loss.mean() 

@jax.jit
def train_step(state_q, key):
    grad_fn = jax.value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_fn(state_q.params, key)
    state_q = state_q.apply_gradients(grads=grads)
    return state_q, loss

# training steps
key, loc_key = jax.random.split(key)
state_q, loss = train_step(state_q, loc_key)

loss_plot = []
for i in trange(n_iterations):
    key, loc_key = jax.random.split(key)
    state_q, loss = train_step(state_q, loc_key)
    # TODO: change plot and print 
    print (f"loss: {jnp.log10(loss):0.3f}")
    loss_plot.append(jnp.log10(loss))

plt.plot(loss_plot)
plt.savefig(f"{savedir}/loss.png")
plt.clf()

# inference steps
# plot q process
BS = 100
t = T*jnp.linspace(0,1,BS).reshape((-1,1))
# plt.contourf(x,y,z,50)
key, path_key = jax.random.split(key)
eps = jax.random.normal(path_key, [BS, 66])
mu_t, sigma_t = state_q.apply_fn(state_q.params, t)
samples = mu_t + sigma_t*eps
# BS, 66
paths = jax.device_get(samples.reshape(1, BS, 22, 3))

# helper function to plot things in 2D (the scatter plot and trajectory)
PlotPathsAlanine(paths, jax.device_get(B[0].reshape(22, 3)), f"{savedir}/sample_q")

# helper function to save in PDB (can be visualized by PyMol)
trajs = None
for i in range(BS):
    traj = md.load_pdb('./files/AD_c7eq.pdb')
    traj.xyz = paths[0][i]
    if i == 0:
        trajs = traj
    else:
        trajs = trajs.join(traj)
trajs.save(f'{savedir}/save_q.pdb')

# inference process
# u_t process
Ux_fn = lambda _x: U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters).sum()
Ux_fn = jax.vmap(Ux_fn)
dUdx_fn = jax.grad(lambda _x: U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters).sum())
dUdx_fn = jax.vmap(dUdx_fn)

mu_t = lambda _t: state_q.apply_fn(state_q.params, _t)[0]
sigma_t = lambda _t: state_q.apply_fn(state_q.params, _t)[1]
def dmudt(_t):
    _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0), argnums=0)
    return _dmudt(_t).squeeze().T
dsigmadt = jax.grad(lambda _t: sigma_t(_t).sum(), argnums=0)
u_t_det = jax.jit(lambda _t, _x: dmudt(_t) + dsigmadt(_t)/sigma_t(_t)*(_x-mu_t(_t)))
u_t_stoch = jax.jit(lambda _t, _x: dmudt(_t) + (dsigmadt(_t)/sigma_t(_t)-0.5)*(_x-mu_t(_t)))

# TODO: batch size is set to be 1 to look at 1 trajectory at a time
BS = 1
N = int(T // dt)
x_t_det = jnp.ones((BS,N+1,66))*jnp.expand_dims(A[:BS], axis=1)
x_t_stoch = jnp.ones((BS,N+1,66))*jnp.expand_dims(A[:BS], axis=1)
t = jnp.zeros((BS,1))
energy_plot_det, energy_plot_stoch = [], []
force_plot_det, force_plot_stoch = [], []
noise_plot = []
Ut_plot = []
for i in range(N):
    key, loc_key = jax.random.split(key)
    eps = jax.random.normal(key, shape=(BS,66))
    # deterministic
    Ut = u_t_det(t, x_t_det[:,i,:])
    dx = dt*Ut
    x_t_det = x_t_det.at[:,i+1,:].set(x_t_det[:,i,:] + dx)
    Ux = Ux_fn(x_t_det[:,i,:])
    # plot energy on deterministic path
    energy_plot_det.append(jnp.log10(Ux))
    # plot force norm on deterministic path
    dUdx = jnp.clip(dUdx_fn(x_t_det[:,i,:]).reshape(BS,22,3), -clip_value, clip_value)/mass.reshape(1, 22, 3)
    norm_dUdx = (dUdx ** 2).sum(axis=-1).mean()
    force_plot_det.append(jnp.log10(norm_dUdx))
    # plot u_t norm on deterministic path
    norm_Ut = (Ut ** 2).sum(axis=-1).mean()
    Ut_plot.append(jnp.log10(norm_Ut))
    # plot sigma * eps / sigma_t term norm on deterministic path
    noise_term = (((sigma.reshape(1, 22, 3)**2)*eps.reshape(1, 22, 3)/sigma_t(t)) ** 2).sum(axis=-1).mean()
    noise_plot.append(jnp.log10(noise_term))

    # stochastic 
    Ut = u_t_stoch(t, x_t_stoch[:,i,:])
    dx = dt*Ut+jnp.sqrt(dt)*sigma_t(t)*eps
    x_t_stoch = x_t_stoch.at[:,i+1,:].set(x_t_stoch[:,i,:] + dx)
    # plot enegy on stochastic path
    Ux = Ux_fn(x_t_stoch[:,i,:])
    energy_plot_stoch.append(jnp.log10(Ux))
    # plot force norm on stochastic path
    dUdx = dUdx_fn(x_t_stoch[:,i,:]).reshape(BS,22,3)
    norm_dUdx = (dUdx ** 2).sum(axis=-1).mean()
    force_plot_stoch.append(jnp.log10(norm_dUdx))
    # increment step
    t += dt

plt.plot(energy_plot_det)
plt.savefig(f"{savedir}/energy_det.png")
plt.clf()

plt.plot(energy_plot_stoch)
plt.savefig(f"{savedir}/energy_stoch.png")
plt.clf()

plt.plot(noise_plot)
plt.savefig(f"{savedir}/noise_plot_det.png")
plt.clf()

plt.plot(force_plot_det)
plt.savefig(f"{savedir}/force_det.png")
plt.clf()

plt.plot(force_plot_stoch)
plt.savefig(f"{savedir}/force_stoch.png")
plt.clf()

plt.plot(Ut_plot)
plt.savefig(f"{savedir}/Ut_det.png")
plt.clf()

# plot the deterministic u_t path
paths = jax.device_get(x_t_det.reshape(BS, N+1, 22, 3))[0:1]
PlotPathsAlanine(paths, jax.device_get(B[0].reshape(22, 3)), f"{savedir}/learned_vector_det")

trajs = None
for i in range(N+1):
    traj = md.load_pdb('./files/AD_c7eq.pdb')
    traj.xyz = paths[0][i]
    if i == 0:
        trajs = traj
    else:
        trajs = trajs.join(traj)
trajs.save(f'{savedir}/save_vector_det.pdb')

# plot the deterministic u_t path
paths = jax.device_get(x_t_stoch.reshape(BS, N+1, 22, 3))[0:1]
PlotPathsAlanine(paths, jax.device_get(B[0].reshape(22, 3)), f"{savedir}/learned_vector_stoch")

trajs = None
for i in range(N+1):
    traj = md.load_pdb('./files/AD_c7eq.pdb')
    traj.xyz = paths[0][i]
    if i == 0:
        trajs = traj
    else:
        trajs = trajs.join(traj)
trajs.save(f'{savedir}/save_vector_stoch.pdb')