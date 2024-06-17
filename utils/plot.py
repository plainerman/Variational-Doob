import matplotlib.pyplot as plt


def show_or_save_fig(save_dir, name):
    if save_dir is not None:
        plt.savefig(f'{save_dir}/{name}', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()
