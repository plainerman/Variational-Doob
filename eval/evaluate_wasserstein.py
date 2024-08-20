import ot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os

if __name__ == '__main__':
    two_way_shooting = np.load('./out/baselines/mueller/paths-two-way-shooting.npy', allow_pickle=True)
    two_way_shooting = two_way_shooting.squeeze(2)
    ours = np.load('./out/toy/mueller_single_gaussian/stochastic_paths.npy')

    assert two_way_shooting.shape == ours.shape, f'Shapes do not match: {two_way_shooting.shape} vs {ours.shape}'

    savedir = './out/evaluation/mueller/'
    os.makedirs(savedir, exist_ok=True)

    wasserstein = []
    for t in trange(ours.shape[1]):
        cur_ground_truth = np.array(two_way_shooting[:, t, :], dtype=np.float64)
        cur_ours = np.array(ours[:, t, :], dtype=np.float64)

        M = ot.dist(cur_ground_truth, cur_ours, metric='euclidean')
        w1 = ot.emd2([], [], M)
        wasserstein.append(w1)

    wasserstein = np.array(wasserstein)
    print('Median Wasserstein:', np.median(wasserstein))
    print('Mean Wasserstein:', np.mean(wasserstein))
    print('Std Wasserstein:', np.std(wasserstein))
    print('Max Wasserstein:', np.max(wasserstein))
    print('Min Wasserstein:', np.min(wasserstein))

    plt.plot(wasserstein)
    plt.xlabel(r'$t$')
    plt.ylabel('Wasserstein W1 Distance')
    plt.savefig(f'{savedir}/wasserstein.pdf', bbox_inches='tight')
    plt.clf()

    print()
