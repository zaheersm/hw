import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(model_name, game='breakout', xlim=None, ylim=None, percentiles=[1, 5, 10], colors=['red', 'orange', 'green']):
    t = np.load('data/output/mini_atari/{}/gan_eval/{}/0_run/0_param_setting/true_beliefs_3907.npy'.format(game, model_name))
    f = np.load('data/output/mini_atari/{}/gan_eval/{}/0_run/0_param_setting/false_beliefs_3907.npy'.format(game, model_name))

    d = t - f

    plt.scatter(f, d, marker='.', s=0.01)
    for k, p in enumerate(percentiles):
        thresh = np.percentile(t, p)
        plt.axvline(x=thresh, color=colors[k], linestyle='--')
        plt.text(thresh + 0.01, 4, 'p={}'.format(p), color=colors[k], rotation='90')

    plt.xlabel("D(s, a, G(s, a))")
    plt.ylabel("D(s, a, s') - D(s, a, G(s, a))")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig('experiment/plots/scatters/{}/{}.png'.format(game, model_name))
    plt.close()


def plot_hist(model_name, game='breakout'):
    t = np.load('data/output/mini_atari/{}/gan_eval/{}/0_run/0_param_setting/true_beliefs_3907.npy'.format(game, model_name))
    f = np.load('data/output/mini_atari/{}/gan_eval/{}/0_run/0_param_setting/false_beliefs_3907.npy'.format(game, model_name))
    plt.figure()
    d = t - f
    plt.subplot(1, 2, 1)
    plt.hist(t, bins=100)
    plt.gca().set_title('True Beliefs')
    plt.ylim((0, 500000))
    plt.xlim((-5, 20))
    plt.subplot(1, 2, 2)
    plt.hist(f, bins=100)
    plt.gca().set_title('False Beliefs')
    plt.ylim((0, 500000))
    plt.xlim((-5, 20))
    plt.savefig('experiment/plots/scatters/space_invaders/hist_{}.png'.format(model_name))
    plt.close()


# plot_scatter('m1_best_20k', xlim=(-6, 2), ylim=(-1, 11))
# plot_scatter('m1_best_30k', xlim=(-6, 2), ylim=(-1, 11))
# plot_scatter('m1_best_50k', xlim=(-6, 2), ylim=(-1, 11))
# plot_scatter('m1_best_80k', xlim=(-6, 2), ylim=(-1, 11))
# plot_scatter('m2_best_20k', xlim=(-6, 2), ylim=(-1, 11))
# plot_scatter('m2_best_30k', xlim=(-6, 2), ylim=(-1, 11))
# plot_scatter('m2_best_50k', xlim=(-6, 2), ylim=(-1, 11))
# plot_scatter('m2_best_80k', xlim=(-6, 2), ylim=(-1, 11))
# plot_scatter('m1_best_v2_disc_120k', game='space_invaders')
# plot_scatter('m1_best_v3_disc_120k', game='space_invaders')
# plot_scatter('m1_best_v4_disc_120k', game='space_invaders')
# plot_scatter('sweep_36', game='space_invaders')
plot_scatter('m1_best_v6_disc_120k', game='space_invaders')

# print("plotting hist")
#
# plot_hist('m1_best_v2_disc_120k', game='space_invaders')
# plot_hist('m1_best_v3_disc_120k', game='space_invaders')
# plot_hist('m1_best_v4_disc_120k', game='space_invaders')
# plot_hist('m1_best_v5_disc_120k', game='space_invaders')
# plot_hist('sagan_666', game='space_invaders')
# plot_hist('sweep_36', game='space_invaders')
# plot_hist('m1_best_v6_disc_120k', game='space_invaders')


