import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage 
import pandas as pd
import reprlib
from pathlib import Path
from pathml.core import SlideData


def plot_composite_segmentation(tile, mask_name, channel_names, 
                                dapi=None, red=None, green=None, blue=None,
                                save_dir=None):
    """
    Overlay three channels on each other.
    Uses adaptive histogram equalization on each channel independently
        
    tile (pathml.Tile): Tile containing image and masks
    mask_name (str): Name of mask to use for plotting segmentations
    channel_names (list): list of channel names, in same order as channels in tile.image 
    dapi (int): index of DAPI channel, to be plotted in grey
    red (int): index of channel to use for red
    green (int): index of channel to use for green
    blue (int): index of channel to use for blue
    save_dir (str): Directory to optionally save output image
    """
    
    # convenience zero array
    _zero = np.zeros((tile.shape[0], tile.shape[1]), dtype=np.uint16)
    
    # get mask and cell boundaries
    mask = tile.masks[mask_name].astype(np.uint16).squeeze()
    boundaries = skimage.segmentation.find_boundaries(mask, connectivity = 1, mode = "inner")
    # this step thins out boundaries that are >1 px wide
    boundaries = skimage.morphology.thin(boundaries)
    boundaries_rgb = np.dstack([boundaries, boundaries, boundaries])   
        
    if dapi is not None:
        dapi_im = tile.image[..., dapi].astype(np.uint16)
        dapi_norm = skimage.exposure.equalize_adapthist(dapi_im, clip_limit=0.03)
        dapi_norm[dapi_norm > 0.2] = 0.2
        dapi_rgb = np.dstack([dapi_norm, dapi_norm, dapi_norm])
    else:
        dapi_rgb = np.dstack([_zero, _zero, _zero])    
    
    if red is not None:
        red_im = tile.image[..., red].astype(np.uint16)
        red_im = skimage.exposure.equalize_adapthist(red_im, clip_limit=0.03)
        red_rgb = np.dstack([red_im, _zero, _zero])
    else:
        red_rgb = np.dstack([_zero, _zero, _zero])
    
    if green is not None:
        green_im = tile.image[..., green].astype(np.uint16)
        green_im = skimage.exposure.equalize_adapthist(green_im, clip_limit=0.03)
        green_rgb = np.dstack([_zero, green_im, _zero])
    else:
        green_rgb = np.dstack([_zero, _zero, _zero])
        
    if blue is not None:
        blue_im = tile.image[..., blue].astype(np.uint16)
        blue_im = skimage.exposure.equalize_adapthist(blue_im, clip_limit=0.03)
        blue_rgb = np.dstack([_zero, _zero, blue_im])
    else:
        blue_rgb = np.dstack([_zero, _zero, _zero])
    
    rgb_merged = red_rgb + green_rgb + blue_rgb
    rgb_merged[rgb_merged > 0.05] += 0.2 # make colors brighter
    rgb_merged += dapi_rgb

    rgb_merged = np.clip(rgb_merged, a_min=None, a_max=1)
            
    rgb_merged[boundaries_rgb] = 0.7
    
    channel_names = [channel_names[ix] if ix else "None" for ix in [red, green, blue]]
        
    # What size does the figure need to be in inches to fit the image?
    dpi = 200
    figsize = tile.shape[1] / float(dpi), tile.shape[0] / float(dpi)
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(rgb_merged, interpolation="nearest")
    ax.axis("off")
    
    patches = [mpatches.Patch(color=color_name, label=channel_name)
               for color_name, channel_name in zip(["red", "green", "blue"], channel_names) if channel_name != "None"]
    if dapi is not None:
        patches.append(mpatches.Patch(color="gray", label="DAPI"))
    ax.legend(handles=patches)
    
    if save_dir:
        fig.savefig(Path(save_dir) / f'tile_{tile.coords}.png', 
                    bbox_inches='tight', dpi=dpi)
        plt.close()
        return
    
    plt.show()
