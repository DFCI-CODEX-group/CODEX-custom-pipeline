import numpy as np
import pandas as pd
import skimage
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
from scipy.ndimage.measurements import label
from skimage.morphology import binary_dilation
from skimage.filters import roberts
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from skimage.morphology import disk,dilation
import anndata
import itertools
import pathml
from pathml.preprocessing.transforms import Transform
import cvmask


class CellSegQuantifyMIF(Transform):
    """
    Convert segmented image into anndata.AnnData counts object `AnnData <https://anndata.readthedocs.io/en/latest/>`_,
    using the least square algorithm for spillover correction.
    For each cell, signal are adjusted in a way such that adjusted signals of a cells and contribution of signals 
    due to overlap from meighboring cells sum to unadjusted (uncompensated) signal.

    Should be used with a segmentation mask where zeros are background, and pixels belonging to each of n
    cells are labelled with integers 1 to n.

    Counts objects are used to interface with the Python single cell analysis ecosystem in
    `Scanpy <https://scanpy.readthedocs.io/en/stable/>`_.
    The counts object contains a summary of channel statistics in each cell along with its coordinate.

    Some parts of this implementation are based on the original implementation by the authors:
    https://github.com/michaellee1/CellSeg/blob/67e85001e7b8f62f844e03779079ab843eddd94e/src/cvmask.py

    Args:
        segmentation_mask (str): key indicating which mask to use as label image
        growth (int): Extent of dilation, higher value means larger dilation (and in turn more aggressive
        correction.) Defaults to 3.
        method (str): Standard mask growth or sequential. Standard mask takes into account neighboring mask while
        growing mask, while sequential doesn't. Deafaults to Standard.
        num_neighbors: Required for Standard growth and defaults to 30. Determines the number of neighbors to 
        take into account while creating connectivty graph fora mask in Standard growth.

    References:
        Lee, M.Y., Bedia, J.S., Bhate, S.S. et al. CellSeg: a robust, pre-trained nucleus segmentation 
        and pixel quantification software for highly multiplexed fluorescence images.
        BMC Bioinformatics 23, 46 (2022). https://doi.org/10.1186/s12859-022-04570-9
        
    """
    def __init__(self, segmentation_mask=None,  growth=3, method = 'Standard', num_neighbors = 30):
        
        self.segmentation_mask = segmentation_mask
        self.growth = growth
        self.method = method
        self.num_neighbors = num_neighbors


        
    def F(self, img, segmentation, coords_offset=(0, 0)):
        
        """
        Functional implementation of marker quantification using the REDSEA algorithm

        Args:
            img (np.ndarray): (h, w, n_channels) Input image
            segmentation (np.ndarray): (h, w) segmentation mask. Zeros are background, and pixels belonging to each of n cells
                are labelled with integers, with a line of zeros separating adjacent regions.
                Labels will be relabeled from 1 to n.
            coords_offset (tuple, optional): Coordinates (i, j) used to convert tile-level coordinates to slide-level.
                Defaults to (0, 0) for no offset.

        Returns:
            anndata.AnnData: Counts matrix
        """
        if segmentation.ndim == 3 and segmentation.shape[2] == 1:
            segmentation = segmentation.squeeze(axis = 2)

        assert segmentation.ndim == 2 and segmentation.shape == img.shape[0:2] and img.ndim == 3, \
            f"segmentation of shape {segmentation.shape} does not match image of shape {img.shape}. " \
            f"Must be of shapes (i, j) and (i, j, n_channels), respectively."
        
        assert not np.isinf(tile.image).any(), "There are np.inf values in the array. If this is\
        is due to numeric overflow, you can fix it by removing np.inf by max of the array (max excluding inf.)"
        assert not np.isnan(tile.image).any(), "Array contains NaN value(s)."

        mask_reindexed, mask_counts = CellSegQuantifyMIF.split_cells(segmentation.copy())
     
        mask_dilated = CellSegQuantifyMIF.grow_masks(mask_reindexed, self.growth, self.method, self.num_neighbors)
        counts_cellseg, counts, area = CellSegQuantifyMIF.convert_seg_to_CVMask(segmentation_zero_boundary, img)
        print("CellSeg count obtained")
        ### this part copied from QuantifyMIF
        countsdataframe = regionprops_table(
            label_image = segmentation,
            intensity_image = img,
            properties = [
                "label",
                "coords",
                "centroid",
                "filled_area",
                "euler_number",
            ],
        )
        # populate anndata object
        # i,j are relative to the input image (0 to img.shape). Adding offset converts to slide-level coordinates
        counts = anndata.AnnData(
            X = counts_cellseg,
            obs = [
                tuple([i + coords_offset[0], j + coords_offset[1]])
                for i, j in zip(
                    countsdataframe["centroid-0"], countsdataframe["centroid-1"]
                )
            ],
        )
        counts.obs["label"] = countsdataframe["label"]
        counts.obs = counts.obs.rename(columns = {0: "y", 1: "x"})
        counts.obs["filled_area"] = countsdataframe["filled_area"]
        counts.obs["euler_number"] = countsdataframe["euler_number"]
        try:
            counts.obsm["spatial"] = np.array(counts.obs[["x", "y"]])
        except:
            print("warning: did not log coordinates in obsm")
        print("CellSeg counts saved as anndata")
        return counts

    def apply(self, tile):
        assert isinstance(
            tile, pathml.core.tile.Tile
        ), f"tile is type {type(tile)} but must be pathml.core.tile.Tile"
        assert (
                self.segmentation_mask in tile.masks
        ), f"passed segmentation mask '{self.segmentation_mask}' does not exist for tile {tile}"
        assert (
                tile.slide_type.stain == "Fluor"
        ), f"Tile has slide_type.stain='{tile.slide_type.stain}', but must be 'Fluor'"
        tile.counts = self.F(
            img = tile.image,
            segmentation = tile.masks[self.segmentation_mask].copy(),
            coords_offset = tile.coords,
        )
        

    @staticmethod
    def get_segmentation_zero_boundary(segmentation):
        """
        Takes as input a segmentation mask, and adds a 1px-wide line of zero pixels between labelled regions.
        These zero boundary pixels are relied on in later steps of the algorithm.
        Note that skimage.segmentation.watershed does have a ``watershed_line`` parameter that is supposed to do this,
        but it is buggy (See: https://github.com/scikit-image/scikit-image/issues/6279) so this is a simple
        reimplementation using morphological operations

        Args:
            segmentation (np.ndarray): Segmentation mask. Zeros are background, and pixels belonging to each of n
                cells are labelled with integers 1 to n.

        Returns:
            np.ndarray: Segmentation mask. Zeros are background, and pixels belonging to each of n
                cells are labelled with integers 1 to n, with a line of zeros separating adjacent regions.
        """
        boundaries = skimage.segmentation.find_boundaries(segmentation, connectivity = 1, mode = "inner")
        # this step thins out boundaries that are >1 px wide
        boundaries = skimage.morphology.thin(boundaries)
        # zero out the boundary pixels that we identified
        segmentation[boundaries] = 0
        return segmentation
    
    @staticmethod  
    def convert_seg_to_CVMask(segmentation, img):
        mask = cvmask.CVMask(segmentation)
        compensated, means, areas = mask.compute_channel_means_sums_compensated(img)
        return compensated, means, areas
    
    @staticmethod
    def split_cells(mask):
        '''
        Takes a masked tile and adds a pixel of space between each cell
        Returns: Relabeled mask with no duplicate labels and total number of mask counts 
        '''
        rob_filtered = roberts(mask)>0
        contours = binary_dilation(rob_filtered)
        sep = (mask>0) & ~rob_filtered #subtract edge to separate cells prior to ndi.label
        lab,num_labs = label(sep, output = int)
        return lab, num_labs


    @staticmethod
    def compute_centroids(mask):
    
        """
        The compute_centroids utility function takes a mask image as input (in the form of a 2D NumPy array)
        and computes and returns the centroids of the individual mask labels in the mask image. 
        """
        
        num_masks = len(np.unique(mask)) - 1
        indices = np.where(mask != 0)
        values = mask[indices[0], indices[1]]

        maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1],
                                                        values]))).rename(columns = {0:"x", 1:"y", 2:"id"})
        centroids = maskframe.groupby('id').agg({'x': 'mean', 'y': 'mean'}).to_numpy()
        
        return centroids

     
        
    @staticmethod
    def remove_overlaps_nearest_neighbors(mask, centroids):
        """
        
        This utility functuon maps an overlapping mask region to nearest mask based
        on distance from the centriods of overlapping mask labels.
        """
        final_masks = np.max(mask, axis = 2)
       
        collisions = np.nonzero(np.sum(mask > 0, axis = 2) > 1)
        collision_masks = mask[collisions]
        collision_index = np.nonzero(collision_masks)
        collision_masks = collision_masks[collision_index]
        collision_frame = pd.DataFrame(np.transpose(np.array([collision_index[0],collision_masks]))
                                      ).rename(columns = {0:"collis_idx", 1:"mask_id"})
        
        grouped_frame = collision_frame.groupby('collis_idx')
        for collis_idx, group in grouped_frame:
            collis_pos = np.expand_dims(np.array(
                [collisions[0][collis_idx], collisions[1][collis_idx]]), axis = 0)
            
            prevval = final_masks[collis_pos[0,0], collis_pos[0,1]]
            mask_ids = list(group['mask_id'])
            curr_centroids = np.array([centroids[mask_id - 1] for mask_id in mask_ids])
            dists = cdist(curr_centroids, collis_pos)
            closest_mask = mask_ids[np.argmin(dists)]
            final_masks[collis_pos[0,0], collis_pos[0,1]] = closest_mask
        
        return final_masks
    
    
    @staticmethod
    def compute_boundbox(mask):
        """
        computes the minimum and maximum (x, y)
        coordinates of the bounding boxes for each connected component (mask label) in the mask.
        """
    
        num_masks = len(np.unique(mask)) - 1
        indices = np.where(mask != 0)
        values = mask[indices[0], indices[1]]

        maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1],
                                                        values]))).rename(columns = {0:"y", 1:"x", 2:"id"})
        bb_mins = maskframe.groupby('id').agg({'y': 'min', 'x': 'min'}).to_records(index = False).tolist()
        bb_maxes = maskframe.groupby('id').agg({'y': 'max', 'x': 'max'}).to_records(index = False).tolist()
        return bb_mins, bb_maxes
    
    
    
    @staticmethod
    def grow_masks(masks, growth, method = 'Standard', num_neighbors = 30):

        """
        utility function for expanding masks by applying dilation, 
        using either a standard method based on the centroids of the masks,
        or a sequential method based on expanding each mask individually.
        growth controls amount of dilation to be applied to the masks. 
        num_neighbors is used for Standard method and determines number of 
        nearest neighbors to consider when creating connectivity graph in
        the Standard method.

        This function returns the dilated mask.
        """
        assert method in ['Standard', 'Sequential']


        num_masks = len(np.unique(masks)) - 1

        if method == 'Standard':
            print("Standard growth selected")


            cent_array = CellSegQuantifyMIF.compute_centroids(masks)
            connectivity_matrix = kneighbors_graph(cent_array, num_neighbors).toarray() * np.arange(1, num_masks + 1)
            connectivity_matrix = connectivity_matrix.astype(int)
            labels = {}
            for n in range(num_masks):
                connections = list(connectivity_matrix[n, :])
                connections.remove(0)
                layers_used = [labels[i] for i in connections if i in labels]
                layers_used.sort()
                currlayer = 0
                for layer in layers_used:
                    if currlayer != layer: 
                        break
                    currlayer += 1
                labels[n + 1] = currlayer

            possible_layers = len(list(set(labels.values())))
            label_frame = pd.DataFrame(list(labels.items()), columns = ["maskid", "layer"])
            image_h, image_w = masks.shape
            expanded_masks = np.zeros((image_h, image_w, possible_layers), dtype = int)

            grouped_frame = label_frame.groupby('layer')
            for layer, group in grouped_frame:
                currids = list(group['maskid'])
                masklocs = np.isin(masks, currids)
                expanded_masks[masklocs, layer] = masks[masklocs]

            dilation_mask = disk(1)
            grown_masks = np.copy(expanded_masks)
            for _ in range(growth):
                for i in range(possible_layers):
                    grown_masks[:, :, i] = dilation(grown_masks[:, :, i], dilation_mask)
            return CellSegQuantifyMIF.remove_overlaps_nearest_neighbors(grown_masks,cent_array)

        elif method == 'Sequential':
            print("Sequential growth selected")
            Y, X = masks.shape
            bb_mins, bb_maxes = CellSegQuantifyMIF.compute_boundbox(masks)
            struc = disk(1)
            for _ in range(growth):
                for i in range(num_masks):
                    mins = bb_mins[i]
                    maxes = bb_maxes[i]
                    minY, minX,= mins[0] - 3*growth, mins[1] - 3*growth,
                    maxY, maxX  = maxes[0] + 3*growth, maxes[1] + 3*growth
                    if minX < 0: minX = 0
                    if minY < 0: minY = 0
                    if maxX >= X: maxX = X - 1
                    if maxY >= Y: maxY = Y - 1

                    currreg = masks[minY:maxY, minX:maxX]
                    mask_snippet = (currreg == i + 1)
                    full_snippet = currreg > 0
                    other_masks_snippet = full_snippet ^ mask_snippet
                    dilated_mask = binary_dilation(mask_snippet, struc)
                    final_update = (dilated_mask ^ full_snippet) ^ other_masks_snippet


                    pix_to_update = np.nonzero(final_update)

                    pix_X = np.array([min(j + minX, X) for j in pix_to_update[1]])
                    pix_Y = np.array([min(j + minY, Y) for j in pix_to_update[0]])

                    masks[pix_Y, pix_X] = i + 1

            return masks

    


class FilterEdgeCells(pathml.preprocessing.transforms.Transform):
    """
    Filter out cells from counts matrix which are within certain number of px from edge of tile.
    When using overlapping tiles, this can be used to avoid double-counting cells that occur in the overlap.

    Args:
        edge_distance (int): distance from edge at which cells are to be filtered (e.g. 100px)
        slide_shape (tuple): dimensions of wsi (e.g. slide.shape).
            Used to check whether an individual tile is on the edge or not.
    """
    def __init__(self, edge_distance, slide_shape):
        self.edge_distance = edge_distance
        self.slide_shape = slide_shape

    def apply(self, tile):
        # first check if tile is on the edge
        i, j = tile.coords
        di, dj = tile.shape[0:2]
        edge_top = i == 0
        edge_bottom = i + di > self.slide_shape[0]
        edge_left = j == 0
        edge_right = j + dj > self.slide_shape[1]

        # this logic filters out obs around each edge
        # the or condition in each row keeps edge obs, if it's an edge tile
        counts = tile.counts.copy()
        newcounts = counts[((counts.obs.y > i + self.edge_distance) | edge_top) &
                           ((counts.obs.y < i + tile.shape[0] - self.edge_distance) | edge_bottom) &
                           ((counts.obs.x > j + self.edge_distance) | edge_left) &
                           ((counts.obs.x < j + tile.shape[1] - self.edge_distance) | edge_right)]
        tile.counts = newcounts.copy()




class CheckNACounts(pathml.preprocessing.transforms.Transform):
    """
    Checks is counts matrix has NA values, prints info if so
    Used for debugging
    """
    def apply(self, tile):
        if np.isnan(tile.counts.X).any():
            print(f"Missing values in counts matrix! tile coords: {tile.coords}")
            print(f"\tcounts shape: {tile.counts.shape}")
            print(f"\tTotal NaNs: {np.isnan(tile.counts.X).sum()}")
            print(f"\tRows with NaNs: {np.isnan(tile.counts.X).any(axis = 1).sum()}")
