import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def draw_figure(fig):
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


def show_image(im, win_name='Win'):
    cv2.imshow(win_name, im)
    cv2.waitKey(0)


def show_tensor(a: torch.Tensor, fig_num=None, title=None, range=(None, None), ax=None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))

    if ax is None:
        fig = plt.figure(fig_num)
        plt.tight_layout()
        plt.cla()
        plt.imshow(a_np, vmin=range[0], vmax=range[1])
        plt.axis('off')
        plt.axis('equal')
        if title is not None:
            plt.title(title)
        draw_figure(fig)
    else:
        ax.cla()
        ax.imshow(a_np, vmin=range[0], vmax=range[1])
        ax.set_axis_off()
        ax.axis('equal')
        if title is not None:
            ax.set_title(title)
        draw_figure(plt.gcf())
