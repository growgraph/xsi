import logging

import networkx as nx
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def set_fontsize(ax, fs, title_pos="center", xlabel=None, ylabel=None):
    ax.set_xlabel(ax.get_xlabel() if xlabel is None else xlabel, fontsize=fs)
    ax.set_ylabel(ax.get_ylabel() if ylabel is None else ylabel, fontsize=fs)
    ax.tick_params(axis="x", rotation=45)
    if ax.get_title(title_pos):
        ax.set_title(ax.get_title(title_pos), fontsize=fs, loc=title_pos)
    if ax.get_legend() is not None:
        plt.setp(ax.get_legend().get_title(), fontsize=int(0.9 * fs))
        plt.setp(ax.get_legend().get_texts(), fontsize=int(0.8 * fs))
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
