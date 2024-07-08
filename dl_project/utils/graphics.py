import matplotlib.pyplot as plt
import numpy as np

def show_images_grid(imgs):
    """Shows a list of images in a square grid.

    Parameters
    ----------
    imgs : list of images
    """
    num_images = len(imgs)
    if num_images > 1:
        ncols = int(np.ceil(num_images**0.5))
        nrows = int(np.ceil(num_images / ncols))
        _, axes = plt.subplots(ncols, nrows, figsize=(15,15))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i < num_images:
                ax.imshow(imgs[ax_i], cmap='Greys_r', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
    else:
        fig = plt.subplot(111)
        fig.imshow(imgs[0], cmap='Greys_r', interpolation='nearest')
        fig.set_xticks([])
        fig.set_yticks([])
    plt.show()