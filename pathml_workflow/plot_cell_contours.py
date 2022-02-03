import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathml.core import SlideData
import tqdm


def plot_tile_seg_contours(tile, channel_ix, mask_name):
    """
    Plot cell segmentation contours for a tile

    Args:
        tile (pathml.core.tile.Tile): processed image tile.
        channel_ix: Index of channel to overlay segmentations on.
        mask_name (str): name of mask to plot (e.g. 'cell_segmentation')

    Returns:
        np.ndarray: image with contours overlaid

    Example::
        im_dapi_with_contours = plot_tile_seg_contours(tile, channel_ix=0, mask_name="watershed")
        plt.imshow(im_dapi_with_contours)
        plt.show()
    """
    # get mask
    mask = tile.masks[mask_name]
    # get DAPI image for overlaying
    im = tile.image.astype(np.float64)
    im_channel = im[:, :, channel_ix].copy()
    # scale to [0, 1]
    im_channel = im_channel / 2 ** 16
    im_channel_rgb = np.stack([im_channel, im_channel, im_channel], axis = -1)

    for cell_ix in np.unique(mask):
        if cell_ix > 0:  # skip background pixels
            # get contours for that cell
            cell_mask = np.array(mask == cell_ix, dtype = np.uint8)
            contours, hierarchy = cv2.findContours(cell_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # plot points from contours onto channel image, in green
            for c in contours:
                for p in c.squeeze(1):
                    im_channel_rgb[p[1], p[0], :] = np.array([0, 1, 0])
    return im_channel_rgb


def plot_contours_save_image(h5path_path, channel_ix, mask_name, imagename):
    """
    Plot cell segmentation contours for whole slide, and save output image.

    Args:
        h5path_path: Path to saved h5path file. Must be already processed, with 'cell_segmentation' masks for
            each tile.
        channel_ix: Index of channel to overlay segmentations on.
        mask_name (str): name of mask to plot (e.g. 'cell_segmentation')
        imagename: filename of saved image. Will be appended with '.jpg'.
    """
    slidedata = SlideData(h5path_path)

    canvas = np.zeros(shape = (slidedata.shape[0], slidedata.shape[1], 3), dtype = np.float64)

    for tile in tqdm.tqdm(slidedata.tiles):
        im_dapi_with_contours = plot_tile_seg_contours(tile, channel_ix, mask_name)
        # add to full image
        i, j = tile.coords
        di, dj = tile.shape[:2]
        canvas[i:i+di, j:j+dj, :] = im_dapi_with_contours

    plt.rcParams["figure.figsize"] = (20, 20)
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.title("exp270-102521SM-HNC14")
    plt.savefig(f"{imagename}.jpg", dpi = 500)
    # plt.show()
