import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils


def time_series_to_plot_real(time_series_batch, dpi=35, title=None):

    series = time_series_batch
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    if title:
        ax.set_title(title)
    ax.plot(series[0].numpy())
    plt.close()

    return fig


def time_series_to_plot(time_series_batch, dpi=35, feature_idx=0, n_images_per_row=4, titles=None):
    """Convert a batch of time series to a tensor with a grid of their plots

    Args:
        time_series_batch (Tensor): (batch_size, seq_len, dim) tensor of time series
        dpi (int): dpi of a single image
        feature_idx (int): index of the feature that goes in the plots (the first one by default)
        n_images_per_row (int): number of images per row in the plot
        titles (list of strings): list of titles for the plots

    Output:
        single (channels, width, height)-shaped tensor representing an image
    """
    # Iterates over the time series
    images = []
    for i, series in enumerate(time_series_batch.detach()):
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        if titles:
            ax.set_title(titles[i])
        ax.plot(series[:, feature_idx].numpy())

        plt.close()

        return fig
