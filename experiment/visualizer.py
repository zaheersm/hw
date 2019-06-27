import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class RunLines(object):
    """
    Draws lines with multiple runs along with their error bars
    """
    def __init__(self, path_formatters, num_runs, num_datapoints, labels, parser_func=None, save_path=None, xlabel=None, ylabel=None, interval=500,
                 ylim=None):
        """
        :param path_formatters: list of generic data paths for each line
                                in which run number can be substituted
        :param num_runs: list of number of runs for each algorithm
        :param num_datapoints: number of datapoints to be expected for each line
        :param parser_func: function to be used for parsing the data file
        :param save_path: save_path to store the plot
        """
        assert len(path_formatters) > 0
        assert len(path_formatters) == len(num_runs)
        assert len(path_formatters) == len(num_datapoints)
        assert len(path_formatters) == len(labels)
        self.path_formatters = path_formatters
        self.num_runs = num_runs
        self.num_datapoints = num_datapoints
        self.parser_func = parser_func
        self.save_path = save_path
        self.labels = labels
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.interval = interval
        self.ylim = ylim

    def draw(self):
        sns.set(style="darkgrid")
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        # colormap = plt.cm.nipy_spectral  # I suggest to use nipy_spectral, Set1,Paired
        colormap = plt.get_cmap('jet')
        # ax1.set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(self.path_formatters))])
        for idx, pf in enumerate(self.path_formatters):
            nr = self.num_runs[idx]
            nd = self.num_datapoints[idx]
            label = self.labels[idx]
            lines = None
            for run in range(nr):
                path = pf.format(run)
                line = self.parser_func(path, nd, self.interval)
                if line is not None:
                    lines = np.concatenate([lines, np.array([line])], axis=0) if lines is not None else np.array([line])
            try:
                mean = np.nanmean(lines, axis=0)

                std = np.nanstd(lines, axis=0)
                ax1.fill_between(range(nd // self.interval), mean - std, mean + std,
                                 alpha=0.1)
                ax1.plot(range(nd // self.interval), mean, label=label, linewidth=2.0)
            except:
                raise
        # labels = map(lambda x: str(int((x * 10000 / 1000))) + 'K', range(0, 5000, 100))
        # plt.xticks(range(0, 1001, 100), labels)
        if self.ylim is not None: ax1.set_ylim(self.ylim[0], self.ylim[1])
        ax1.legend(loc="best", frameon=False)
        if self.xlabel is not None:
            ax1.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax1.set_ylabel(self.ylabel)
        fig1.savefig(self.save_path, format='pdf')
        plt.close(fig1)


class RunLinesIndividual(object):
    """
    Draws lines with multiple runs along with their error bars
    """
    def __init__(self, path_formatters, num_runs, num_datapoints, labels, parser_func=None, save_path=None, xlabel=None, ylabel=None, interval=500):
        """
        :param path_formatters: list of generic data paths for each line
                                in which run number can be substituted
        :param num_runs: list of number of runs for each algorithm
        :param num_datapoints: number of datapoints to be expected for each line
        :param parser_func: function to be used for parsing the data file
        :param save_path: save_path to store the plot
        """
        assert len(path_formatters) > 0
        assert len(path_formatters) == len(num_runs)
        assert len(path_formatters) == len(num_datapoints)
        assert len(path_formatters) == len(labels)
        self.path_formatters = path_formatters
        self.num_runs = num_runs
        self.num_datapoints = num_datapoints
        self.parser_func = parser_func
        self.save_path = save_path
        self.labels = labels
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.interval = interval

    def draw(self):
        sns.set(style="darkgrid")
        fig1 = plt.figure()
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
        ax1 = fig1.add_subplot(111)
        # colormap = plt.cm.nipy_spectral  # I suggest to use nipy_spectral, Set1,Paired
        colormap = plt.get_cmap('jet')
        ax1.set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(self.path_formatters))])
        for idx, pf in enumerate(self.path_formatters):
            nr = self.num_runs[idx]
            nd = self.num_datapoints[idx]
            label = self.labels[idx]
            lines = None
            is_label = False
            for run_id, run in enumerate(range(nr)):
                path = pf.format(run)
                line = self.parser_func(path, nd, self.interval)
                if line is not None:
                    lines = np.concatenate([lines, np.array([line])], axis=0) if lines is not None else np.array([line])
                    if not is_label:
                        ax1.plot(range(nd // self.interval), line, label=label, linewidth=1.0, color=colors[run_id],
                                 linestyle=':')
                        is_label = True
                    else:
                        ax1.plot(range(nd // self.interval), line, linewidth=1.0, color=colors[run_id],
                                 linestyle=':')
        # ax1.set_ylim(-200, 300)
        ax1.legend(loc="best", frameon=False)
        if self.xlabel is not None:
            ax1.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax1.set_ylabel(self.ylabel)
        fig1.savefig(self.save_path)
        plt.close(fig1)







