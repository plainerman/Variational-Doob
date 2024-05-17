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

num_mixtures = 3
trainable_weights = False

class MLPq(nn.Module):
    @nn.compact
    def __call__(self, t):
        # v1 [128, 256, 512, 1024, 512, 512]
        # v2 
        # v3 skip connection
        # future version GNN
        t = t/T
        h = nn.Dense(512)(t-0.5)
        h = nn.swish(h)
        h = nn.Dense(512)(h)
        h = nn.swish(h)
        h = nn.Dense(512)(h)
        h = nn.swish(h)
        h = nn.Dense(512)(h)
        h = nn.swish(h)
        h = nn.Dense(512)(h)
        h = nn.swish(h)
        h = nn.Dense(264 * num_mixtures)(h)
        h = h.reshape(-1, num_mixtures, 264)
        t = t[:, None, :]
        # TODO: ResNet
        # h1 = nn.Dense(1024)(t-0.5)
        # h1 = nn.swish(h1)
        # h2 = nn.Dense(1024)(h1)
        # h2 = h2 + h1
        # h2 = nn.swish(h2)
        # h3 = nn.Dense(1024)(h2)
        # h3 = h3 + h2
        # h3 = nn.swish(h3)
        # h = nn.Dense(264)(h3)
        mu = (1-t)*A[:BS] + t*B[:BS] + (1-t)*t*h[:,:,:132]
        sigma = (1-t)*1e-5 + t*1e-5 + (1-t)*t*jnp.exp(h[:,:,132:])
        weights = self.param('w_logits', nn.initializers.zeros_init(), (num_mixtures,)) if trainable_weights else jnp.zeros(num_mixtures)
        return mu, sigma, weights

def remove_center_of_mass(x):
    x = x.reshape(-1, 22, 3)
    x = x - x.mean(axis=1, keepdims=True)
    x = x.reshape(-1, 66)
    return x

# read initial and target files 
# init_pdb = app.PDBFile("/mnt/beegfs/bulk/mirror/yuanqi/tps/wlf/AD_c7eq_aligned.pdb")
# target_pdb = app.PDBFile("./files/AD_c7ax.pdb")
init_pdb = app.PDBFile("./files/AD_A.pdb")
target_pdb = app.PDBFile("./files/AD_B.pdb")

# Hyperparameters
# T = 5e-5 # 1e-12 5e-5
# dt = 1e-7 # 1e-15 1e-7
T = 1
dt = 1e-3
BS = 256
lr = 1e-4
clip_value = 1e8 # clip force
energy_weight = 1e-0 #1e-4 # scale down energy
n_iterations = 100000

# working dir
savedir = f"models_variational_gp_2nd_mixture/alanine_clip{clip_value}_weight{energy_weight}_lr{lr}_sigma{1e-5}_dt{dt}_iterations{n_iterations}_removeCoM_largenetwork_512_fixtarget_mixtures{num_mixtures}"
os.makedirs(savedir, exist_ok=True)
# Boltzmann constant
# temp = 298.15
temp = 300
kbT = 1.380649 * 6.02214076 * 1e-3 * temp
# Construct the mass matrix
mass = [a.element.mass.value_in_unit(unit.dalton) for a in init_pdb.topology.atoms()]
new_mass = []
for mass_ in mass:
    for _ in range(3):
        new_mass.append(mass_)
mass = jnp.array(new_mass)
# Obtain sigma, gamma is by default 1
xi = jnp.sqrt(2 * kbT / mass)
xi_pos = jnp.zeros_like(xi) + 1e-5
xi = jnp.concatenate((xi_pos, xi), axis=-1)

# Initial and target shape [BS, 66]
A = jnp.array(init_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(1, 1, -1)
A = remove_center_of_mass(A)
A = jnp.concatenate((A, jnp.zeros_like(A)), axis=1)
A = jnp.tile(A, (BS, num_mixtures, 1))
B = jnp.array(target_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(1, 1, -1)
B = remove_center_of_mass(B)
B = jnp.concatenate((B, jnp.zeros_like(B)), axis=1)
B = jnp.tile(B, (BS, num_mixtures, 1))
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
    eps = jax.random.normal(key[1], [BS,1,132])
    mu_t = lambda _t: state_q.apply_fn(params_q, _t)[0]
    sigma_t = lambda _t: state_q.apply_fn(params_q, _t)[1]
    w_logits = state_q.apply_fn(params_q, t)[2]

    i = jax.random.categorical(key[2], w_logits, shape=(BS,))

    def dmudt(_t):
        _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0).T)
        return _dmudt(_t).squeeze().T
    def dsigmadt(_t):
        _dsigmadt = jax.jacrev(lambda _t: sigma_t(_t).sum(0).T)
        return _dsigmadt(_t).squeeze().T

    dUdx_fn = jax.grad(lambda _x: energy_weight * U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters).sum())
    dUdx_fn = jax.vmap(dUdx_fn)

    def drift(_x):
        return jnp.hstack([_x[:,66:], -jnp.clip(dUdx_fn(_x[:,:66]), -clip_value, clip_value)/(mass)])

    def v_t(_eps, _t, _i, _w_logits):
        _mu_t = mu_t(_t)
        _sigma_t = sigma_t(_t)
        # we only sample _x from one of the Gaussians
        _x = _mu_t[jnp.arange(BS), _i, None] + _sigma_t[jnp.arange(BS), _i, None] * _eps
        _drift = drift(_x.squeeze(1))

        log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
        relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

        # print (relative_mixture_weights.shape, _sigma_t.shape, _x.shape, _mu_t.shape, (_x - _mu_t).shape)
        # print ((relative_mixture_weights / (_sigma_t ** 2)).shape)
        log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)

        # print (dsigmadt(_t).shape, dmudt(_t).shape)
        u_t = (relative_mixture_weights * (1 / _sigma_t * dsigmadt(_t) * (_x - _mu_t) + dmudt(_t))).sum(axis=1)

        return u_t - _drift + 0.5 * (xi ** 2) * log_q_t
    loss = 0.5 * ((v_t(eps, t, i, w_logits) / xi) ** 2).sum(-1, keepdims=True)
    # loss = v_t(eps, t)
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
    # exit(0)
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
eps = jax.random.normal(path_key, [BS, 1, 132])
mu_t, sigma_t, w_logits = state_q.apply_fn(state_q.params, t)
w = jax.nn.softmax(w_logits)[None, :, None]
samples = (w * (mu_t + sigma_t*eps)).sum(axis=1)
samples = samples[:, :66]

plt.plot(sigma_t[:, 0])
plt.savefig(f"{savedir}/sigma_t.png")
plt.clf()

# BS, 66
paths = jax.device_get(samples.reshape(1, BS, 22, 3))

# helper function to plot things in 2D (the scatter plot and trajectory)
PlotPathsAlanine(paths, jax.device_get(B[0,0,:66].reshape(22, 3)), f"{savedir}/sample_q")

# helper function to save in PDB (can be visualized by PyMol)
trajs = None
for i in range(BS):
    traj = md.load_pdb('/mnt/beegfs/bulk/mirror/yuanqi/tps/SOCTransitionPaths/potentials/files/AD_c7eq.pdb')
    traj.xyz = paths[0][i]
    if i == 0:
        trajs = traj
    else:
        trajs = trajs.join(traj)
trajs.save(f'{savedir}/save_sample_q.pdb')

# save the mean
samples = (w * (mu_t)).sum(axis=1)
samples = samples[:, :66]
# samples = mu_t[:, :66]
# BS, 66
paths = jax.device_get(samples.reshape(1, BS, 22, 3))

# helper function to plot things in 2D (the scatter plot and trajectory)
PlotPathsAlanine(paths, jax.device_get(B[0,0, :66].reshape(22, 3)), f"{savedir}/sample_mean_q")

# helper function to save in PDB (can be visualized by PyMol)
trajs = None
for i in range(BS):
    traj = md.load_pdb('/mnt/beegfs/bulk/mirror/yuanqi/tps/SOCTransitionPaths/potentials/files/AD_c7eq.pdb')
    traj.xyz = paths[0][i]
    if i == 0:
        trajs = traj
    else:
        trajs = trajs.join(traj)
trajs.save(f'{savedir}/save_mean_q.pdb')

# plot weighted mixture path


# inference process
# u_t process
Ux_fn = lambda _x: energy_weight * U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters).sum()
Ux_fn = jax.vmap(Ux_fn)
dUdx_fn = jax.grad(lambda _x: energy_weight * U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters).sum())
dUdx_fn = jax.vmap(dUdx_fn)

mu_t = lambda _t: state_q.apply_fn(state_q.params, _t)[0]
sigma_t = lambda _t: state_q.apply_fn(state_q.params, _t)[1]
w_logits = lambda _t: state_q.apply_fn(state_q.params, _t)[2]
def dmudt(_t):
    _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0).T, argnums=0)
    return _dmudt(_t).squeeze().T
def dsigmadt(_t):
    _dsigmadt = jax.jacrev(lambda _t: sigma_t(_t).sum(0).T)
    return _dsigmadt(_t).squeeze().T
@jax.jit
def u_t_det(_t, _x):
    _mu_t = mu_t(_t)
    _sigma_t = sigma_t(_t)
    _w_logits = w_logits(_t)
    _x = _x[:, None, :]

    log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
    relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

    return (relative_mixture_weights * (1 / _sigma_t * dsigmadt(_t) * (_x - _mu_t) + dmudt(_t))).sum(axis=1)
# u_t_det = jax.jit(lambda _t, _x: dmudt(_t) + dsigmadt(_t)/sigma_t(_t)*(_x-mu_t(_t)))
# u_t_stoch = jax.jit(lambda _t, _x: dmudt(_t) + (dsigmadt(_t)/sigma_t(_t) - 0.5*(xi/sigma_t(_t))**2)*(_x-mu_t(_t)))

# TODO: batch size is set to be 1 to look at 1 trajectory at a time
BS = 1
N = int(T // dt)
x_t_det = jnp.ones((BS,N+1,132))*A[:BS, 0:1]
x_t_stoch = jnp.ones((BS,N+1,132))*A[:BS, 0:1]
key, loc_key = jax.random.split(key)
eps = jax.random.normal(key, shape=(BS,132))
# TODO: remove noise
print (sigma_t(jnp.zeros((BS,1))).shape)
x_t_det = x_t_det.at[:, 0, :].set(x_t_det[:, 0, :] + sigma_t(jnp.zeros((BS,1)))[:, 0, :]*eps)
key, loc_key = jax.random.split(key)
eps = jax.random.normal(key, shape=(BS,132))
# TODO: remove noise
x_t_stoch = x_t_stoch.at[:, 0, :].set(x_t_stoch[:, 0, :] + sigma_t(jnp.zeros((BS,1)))[:, 0, :]*eps)
t = jnp.zeros((BS,1))
energy_plot_det, energy_plot_stoch = [], []
force_plot_det, force_plot_stoch = [], []
noise_plot = []
Ut_plot = []
for i in trange(N):
    key, loc_key = jax.random.split(key)
    eps = jax.random.normal(key, shape=(BS,132))
    # deterministic
    Ut = u_t_det(t, x_t_det[:,i,:])
    dx = dt*Ut
    new_x_t_det = x_t_det[:,i,:] + dx
    x_t_det = x_t_det.at[:,i+1,:].set(new_x_t_det)
    Ux = Ux_fn(x_t_det[:,i,:66])
    # print (Ux)
    # plot energy on deterministic path
    energy_plot_det.append(jnp.log10(Ux))
    # plot force norm on deterministic path
    dUdx = jnp.clip(dUdx_fn(x_t_det[:,i,:66]).reshape(BS,22,3), -clip_value, clip_value)/mass.reshape(1, 22, 3)
    # print (dUdx)
    norm_dUdx = (dUdx ** 2).sum(axis=-1).mean()
    force_plot_det.append(jnp.log10(norm_dUdx))
    # plot u_t norm on deterministic path
    norm_Ut = (Ut ** 2).sum(axis=-1).mean()
    Ut_plot.append(jnp.log10(norm_Ut))
    # plot sigma * eps / sigma_t term norm on deterministic path
    # noise_term = (((xi.reshape(1, 22, 3)**2)*eps.reshape(1, 22, 3)/sigma_t(t)) ** 2).sum(axis=-1).mean()
    # noise_plot.append(jnp.log10(noise_term))

    # # stochastic 
    # Ut = u_t_stoch(t, x_t_stoch[:,i,:])
    # dx = dt*Ut+jnp.sqrt(dt)*xi*eps
    # new_x_t_stoch = x_t_stoch[:,i,:] + dx
    # x_t_stoch = x_t_stoch.at[:,i+1,:].set(new_x_t_stoch)
    # # plot enegy on stochastic path
    # Ux = Ux_fn(x_t_stoch[:,i,:66])
    # energy_plot_stoch.append(jnp.log10(Ux))
    # # plot force norm on stochastic path
    # dUdx = dUdx_fn(x_t_stoch[:,i,:66]).reshape(BS,22,3)
    # norm_dUdx = (dUdx ** 2).sum(axis=-1).mean()
    # force_plot_stoch.append(jnp.log10(norm_dUdx))
    # increment step
    t += dt

plt.plot(energy_plot_det)
plt.savefig(f"{savedir}/energy_det.png")
plt.clf()

# plt.plot(energy_plot_stoch)
# plt.savefig(f"{savedir}/energy_stoch.png")
# plt.clf()

# plt.plot(noise_plot)
# plt.savefig(f"{savedir}/noise_plot_det.png")
# plt.clf()

plt.plot(force_plot_det)
plt.savefig(f"{savedir}/force_det.png")
plt.clf()

# plt.plot(force_plot_stoch)
# plt.savefig(f"{savedir}/force_stoch.png")
# plt.clf()

plt.plot(Ut_plot)
plt.savefig(f"{savedir}/Ut_det.png")
plt.clf()

print (x_t_det.shape, x_t_stoch.shape)

# plot the deterministic u_t path
paths = jax.device_get(x_t_det[:,:,:66].reshape(BS, N+1, 22, 3))[0:1]
PlotPathsAlanine(paths, jax.device_get(B[0][0,:66].reshape(22, 3)), f"{savedir}/learned_vector_det")

trajs = None
for i in range(N+1):
    traj = md.load_pdb('/mnt/beegfs/bulk/mirror/yuanqi/tps/SOCTransitionPaths/potentials/files/AD_c7eq.pdb')
    traj.xyz = paths[0][i]
    if i == 0:
        trajs = traj
    else:
        trajs = trajs.join(traj)
trajs.save(f'{savedir}/save_vector_det.pdb')

# # plot the deterministic u_t path
# paths = jax.device_get(x_t_stoch[:,:,:66].reshape(BS, N+1, 22, 3))[0:1]
# PlotPathsAlanine(paths, jax.device_get(B[0][:66].reshape(22, 3)), f"{savedir}/learned_vector_stoch")

# trajs = None
# for i in range(N+1):
#     traj = md.load_pdb('/mnt/beegfs/bulk/mirror/yuanqi/tps/SOCTransitionPaths/potentials/files/AD_c7eq.pdb')
#     traj.xyz = paths[0][i]
#     if i == 0:
#         trajs = traj
#     else:
#         trajs = trajs.join(traj)
# trajs.save(f'{savedir}/save_vector_stoch.pdb')