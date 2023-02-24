import numpy as np
import pandas as pd
import skimage
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
from scipy.ndimage.measurements import label
from skimage.morphology import binary_dilation
from skimage.filters import roberts
from pathml.preprocessing.transforms import Transform
from skimage.filters import roberts
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from skimage.morphology import disk,dilation
import anndata
import itertools
import pathml
import re



class REDSEAQuantifyMIF(Transform):
    """
    Convert segmented image into anndata.AnnData counts object `AnnData <https://anndata.readthedocs.io/en/latest/>`_,
    using the REinforcement Dynamic Spillover EliminAtion (REDSEA) algorithm for spillover correction.
    For each cell, signal from the boundary pixels is reinforced, while signal from boundary pixels of neighboring
    cells is subtracted.

    Should be used with a segmentation mask where zeros are background, and pixels belonging to each of n
    cells are labelled with integers 1 to n.

    Counts objects are used to interface with the Python single cell analysis ecosystem in
    `Scanpy <https://scanpy.readthedocs.io/en/stable/>`_.
    The counts object contains a summary of channel statistics in each cell along with its coordinate.

    Some parts of this implementation are based on the original implementation by the authors:
    https://github.com/nolanlab/REDSEA/tree/master/code/REDSEApy

    Args:
        segmentation_mask (str): key indicating which mask to use as label image
        element_size (int): width of structuring element used to determine if the pixel is a boundary pixel.
            Number of pixels from center of structuring element to edge (i.e., radius), although this may be calculated
            differently for different elements. Defaults to 2. Increasing it results in more aggressive correction since
            redsea is sensitive to boundary size which increases on increasing element_size. 
        element_shape (str): shape of structuring element used to determine if the pixel is a boundary pixel.
            Supports "diamond", "disk", "square", and "star". Defaults to "diamond".
            See: https://scikit-image.org/docs/dev/auto_examples/numpy_operations/plot_structuring_elements.html
            
        growth (int): Extent of dilation, higher value means larger dilation (and in turn more aggressive
        correction.) Defaults to 3.
        
        method (str): Standard mask growth or sequential. Standard mask takes into account neighboring mask while
        growing mask, while sequential doesn't. Deafaults to Standard.
        
        num_neighbors: Required for Standard growth and defaults to 30. Determines the number of neighbors to 
        take into account while creating connectivty graph fora mask in Standard growth.   
        
        use_actual_overlap_for_reinforcement: True or False; False by default. Specifies whether to use actual overlap
            or whole boundary signals for reinforcement. Note that the paper approximates that the whole boundary area is 
            overlapping with neighboring cells.
            
        alpha:float b/w [0,1]; Defualts to 1. Specifies extent of reinforcement. A higher value results into higher fraction 
            of cell boundary signal being used for reinforcing the signal. Note that paper assumes alpha to be 1 (i.e, uses
            signals on all boundary area for reinforcement.)

    References:
        Bai Y, Zhu B, Rovira-Clave X, Chen H, Markovic M, Chan CN, Su T-H, McIlwain DR, Estes JD,
        Keren L, Nolan GP and Jiang S (2021) Adjacent Cell Marker Lateral Spillover Compensation and Reinforcement
        for Multiplexed Images. Front. Immunol. 12:652631. doi: 10.3389/fimmu.2021.652631
    """
    def __init__(self, segmentation_mask=None, element_size=2, element_shape="diamond",
                growth = 3, method = 'Standard', num_neighbors = 30, use_actual_overlap_for_reinforcement = False, alpha = 1):
        
        
        if element_shape == "diamond":
            self.structuring_element = skimage.morphology.diamond(element_size)
        elif element_shape == "square":
            # square is initialized with full width, so need to convert from width from center used by other elements
            square_size = element_size * 2 + 1
            self.structuring_element = skimage.morphology.square(square_size)
        elif element_shape == "disk":
            self.structuring_element = skimage.morphology.disk(element_size)
        elif element_shape == "star":
            self.structuring_element = skimage.morphology.star(element_size)
        else:
            raise ValueError(f"input element shape {element_shape} not valid")

        self.element_shape = element_shape

        assert self.structuring_element.shape[0] % 2 == 1, \
            f"invalid structuring element shape: {self.structuring_element.shape}. Must have odd dimensions so that" \
            f"element can be centered on a single pixel."
        
        
        self.element_size = int((self.structuring_element.shape[0] - 1) / 2)
        self.segmentation_mask = segmentation_mask
        
        self.growth = growth
        self.method = method
        self.num_neighbors = num_neighbors
        self.use_actual_overlap_for_reinforcement = use_actual_overlap_for_reinforcement
        self.alpha = alpha
        

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

#         segmentation_zero_boundary = self.get_segmentation_zero_boundary(segmentation.copy())
        
        assert not np.isinf(tile.image).any(), "There are np.inf values in the array. If this is\
        is due to numeric overflow, you can fix it by removing np.inf by max of the array (max excluding inf.)"
        assert not np.isnan(tile.image).any(), "Array contains NaN value(s)."

        mask_reindexed, mask_counts = REDSEAQuantifyMIF.split_cells(segmentation.copy())
     
        mask_dilated = REDSEAQuantifyMIF.grow_masks(mask_reindexed, self.growth, self.method, self.num_neighbors)

        counts_redsea, counts_uncomp = self.compute_redsea_counts_matrix(img=img, mask=mask_dilated)

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
            X = counts_redsea,
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
        
        
    def compute_redsea_counts_matrix(self, img, mask):
        """
        Computes counts matrix using REDSEA algorithm

        Args:
            img (np.ndarray): (h, w, n_channels) Input image
            mask (np.ndarray): (h, w) segmentation mask. Zeros are background, and pixels belonging to each of n cells
                are labelled with integers, with a line of zeros separating adjacent regions.
                Labels will be relabeled from 1 to n.
            use_actual_overlap_for_reinforcement: True or False; False by default. Specifies whether to use actual overlap
            or whole boundary signals for reinforcement. Note that the paper approximates that the whole boundary area is 
            overlapping with neighboring cells.
            alpha:float b/w [0,1]; Defualts to 1. Specifies extent of reinforcement. A higher value results into higher fraction 
            of cell boundary signal being used for reinforcing the signal. Note that paper assumes alpha to be 1 (i.e, uses
            signals on all boundary area for reinforcement.)
        Returns:
            np.ndarray: counts_redsea matrix (n_cells, n_channels), the corrected signals
            np.ndarray: counts matrix (n_cells, n_channels), the uncorrected/original signals
        """
        # make sure that labels are 1:n
        # this is relied upon later, when NxN matrix is indexed using cell labels
        mask, _, _ = relabel_sequential(mask)

        labels = np.unique(mask)
        # remove zero label (background pixels)
        labels = [item for item in labels if item != 0]
        n_cells = len(labels)
        # check to make sure that labels are consecutive
        assert n_cells == np.max(mask), "labels must be consecutive ints from 1:n"

        n_channels = img.shape[2]

        # get border mask
        mask_border = REDSEAQuantifyMIF._mask_to_border_mask(
            mask,
            structuring_element = self.structuring_element,
            element_size = self.element_size
        )

        # get cell-cell interaction matrix
        cell_adjecencies, cell_perimeters = REDSEAQuantifyMIF._compute_pairwise_matrix(mask, 
                                                                          structuring_element=self.structuring_element)
        cell_pair_weights = cell_adjecencies/cell_perimeters
        

        counts = np.empty((n_cells, n_channels))
        counts_border = np.empty((n_cells, n_channels))
        cell_sizes = np.empty(n_cells)

        # fill counts matrix and border counts matrix, and cell size
        for label in labels:
            counts_lab = img[mask == label].sum(axis = 0)
            counts_border_lab = img[mask_border == label].sum(axis = 0)

            cell_sizes[label - 1] = np.sum(mask == label)
            counts[label - 1, :] = counts_lab
            counts_border[label - 1, :] = counts_border_lab

        # computes the weighted sum of border counts, to be subtracted
        if self.use_actual_overlap_for_reinforcement:
            # If reinforcing signal based on actual overlap of boundary area
            print("Using exact overlap for reinforcement and subtraction of spillover signal")
            cell_pair_weights_self = cell_adjecencies.sum(axis=1)/cell_perimeters 
            
        # Note that equation 5 in redsea paper is an approximation. The overlapping reagions of the cell won't sum to its 
        # perimeter. # Nonetheless, using only overlapping reason for reinforcement (count_border variable below) results
        # in very little spillover correction. So, we implement the algorithm here as it is in paper, but with a hyperparamter
        # that can let you specify extent of border signals you want to reinforce with. We call this parameter alpha.
        # We also provide an option to get the actual overlap of the cell.
        
            counts_border = cell_pair_weights_self.reshape((-1, 1)) * counts_border
        else:
            # If using the approximation of the paper (equation 5)
            counts_border = counts_border
            
        counts_subtract = cell_pair_weights @ counts_border

        # now apply reinforcement and subtraction
        counts_redsea = counts + self.alpha*counts_border - counts_subtract

        # normalize by cell area
        counts_redsea = np.diag([1 / cell_size for cell_size in cell_sizes]) @ counts_redsea
        counts = np.diag([1 / cell_size for cell_size in cell_sizes]) @ counts

        # clip negative values to 0
        counts_redsea = counts_redsea.clip(0)

        return counts_redsea, counts

    

    @staticmethod
    def _mask_to_border_mask(mask, structuring_element, element_size):
        
        """
        Converts a mask to a border-only mask.
        Loops through all pixels in the input mask, applies structuring element to get adjacent pixels, and
        and keeps the pixel only if a boundary region (i.e., zero pixel) is among the neighbors. Otherwise,
        sets the pixel to zero as a non-boundary pixel.

        Args:
            mask (np.ndarray): (h, w) segmentation mask. Zeros are background, and pixels belonging to each of n
                cells are labelled with unique integers, with a line of zeros separating adjacent regions.
            element_size (int): width of structuring element used to determine if the pixel is a boundary pixel.
                Defaults to 2.
            element_shape (str): shape of structuring element used to determine if the pixel is a boundary pixel.
                Defaults to "diamond" to use ``skimage.morphology.diamond``

        Returns:
            np.ndarray: Segmentation mask of same shape as input, with zeros for non-boundary pixels
        """
        
        assert mask.ndim == 2, f"input mask has shape {mask.shape} but must be (h, w)"

        border_mask = np.copy(mask)

        elem_loc = np.where(structuring_element == 1)
        # offset so that the location indices are centered on the structuring element
        elem_loc_i, elem_loc_j = [c - element_size for c in elem_loc]

        # loop over pixels
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # loop thru structuring element
                border = False
                for di, dj in zip(elem_loc_i, elem_loc_j):
                    if 0 <= i + di <= mask.shape[0] - 1 and 0 <= j + dj <= mask.shape[1] - 1:
                        if mask[i + di, j + dj] == 0:
                            # zero pixel means that it's a boundary pixel
                            # bigger element size --> thicker boundary
                            border = True
                if not border:
                    # in this case, no zero pixels were found, so it's not a border pixel
                    border_mask[i, j] = 0
        return border_mask

    @staticmethod
    def _compute_pairwise_matrix(mask, structuring_element):
        """
        Computes perimeters of each cell, and pairwise compensation values for each pair.
        Uses a structuring element.
        Note that labels are counted starting at 1, but array is indexed starting at 0

        Args:
            mask (np.ndarray): (h, w) segmentation mask. Zeros are background, and pixels belonging to each of n
                cells are labelled with integers 1 to n, with a line of zeros separating adjacent regions.
            structuring_element

        Returns:
            np.ndarray: (n_cells, n_cells) array where the i,jth element gives b_ij / P_j, i.e. the number of boundary pixels
                between cells i and j divided by the total perimeter of cell j.
            np.ndarray: (m_cells, ) array where i_th element contains the perimeter corresponding to the cell i (actually i+1)
            since cell label start at 1, while array indices start at 0.
        """
        labels = np.unique(mask)
        # remove zero label (background pixels)
        labels = [item for item in labels if item != 0]
        n_cells = len(labels)

        # check to make sure that labels are consecutive
        assert n_cells == np.max(mask), "labels must be consecutive ints from 1:n"

        # Add a padding of zeros around outside as well
        # Since zero pixels are used to count cell perimeters, this is necessary otherwise perimeter will be
        # underestimated for cells touching image border
        mask = np.pad(mask, pad_width = 1, mode = 'constant', constant_values = 0)

        cell_perimeters = np.zeros(n_cells)
        cell_adjacencies = np.zeros((n_cells, n_cells))  # cell-cell shared perimeter matrix container

        zero_ind_i, zero_ind_j = np.where(mask == 0)
        
        # loop thru zero pixels
        for i, j in zip(zero_ind_i, zero_ind_j):
            # offsets for a 3x3 structuring element centered on i,j
            adj_labels = []
#             for di, dj in itertools.product([-1, 0, 1], repeat = 2):
            elem_loc = np.where(structuring_element == 1)
        # offset so that the location indices are centered on the structuring element
            elem_loc_i, elem_loc_j = [c - structuring_element.shape[0]//2+1 for c in elem_loc]
            for di, dj in zip(elem_loc_i, elem_loc_j):
                if 0 <= i + di <= mask.shape[0] - 1 and 0 <= j + dj <= mask.shape[1] - 1:
                    label = mask[i + di, j + dj]
                    if label != 0 and label not in adj_labels:
                        adj_labels.append(label)
                        
            # increment perimeter counts
            # need to subtract 1 because labels start at 1 but index starts at 0
            for label in adj_labels:
                cell_perimeters[int(label) - 1] += 1

            # increment shared perimeter counts
            # need to subtract 1 because labels start at 1 but index starts at 0
            for label1, label2 in itertools.permutations(adj_labels, 2):
                cell_adjacencies[label1 - 1, label2 - 1] += 1

        # raise if any of the cell perimeters are zero, since this will introduce NaNs in the next division step
        # also this really shouldn't ever happen, since every region has to have a perimeter>0
        if np.any(cell_perimeters == 0):
            raise ValueError("Cell perimeters in _compute_pairwise_matrix() contains zeros!!")

        # divide to get fraction
#         cell_adjacencies = cell_adjacencies / cell_perimeters

        return cell_adjacencies, cell_perimeters

    
    
    

    
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


            cent_array = REDSEAQuantifyMIF.compute_centroids(masks)
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
            return REDSEAQuantifyMIF.remove_overlaps_nearest_neighbors(grown_masks,cent_array)

        elif method == 'Sequential':
            print("Sequential growth selected")
            Y, X = masks.shape
            bb_mins, bb_maxes = REDSEAQuantifyMIF.compute_boundbox(masks)
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


class MembraneMarkerWatershed(pathml.preprocessing.transforms.Transform):
    """
    Performs marker-controlled watershed segmentation.
    Uses nuclei segmentation as markers. Adds result to tile as "watershed" segmentation mask.
    Wraps ``skimage.segmentation.watershed``

    Args:
        membrane_channel (int): index of marker to use for filling with watershed algorithm
        marker_segmentation_mask (str): name of segmentation mask to use as markers
        watershed_line (bool): If watershed_line is True, a one-pixel wide line separates the regions obtained
          by the watershed algorithm. The line has the label 0. Defaults to True.
        mask_name (str): name for new mask. Defaults to "watershed"
    """
    def __init__(self, membrane_channel, marker_segmentation_mask, watershed_line=True, mask_name="watershed"):
        self.membrane_channel = membrane_channel
        self.marker_segmentation_mask = marker_segmentation_mask
        self.watershed_line = watershed_line
        self.mask_name = mask_name

    def apply(self, tile):
        watershed = skimage.segmentation.watershed(
            image = tile.image[..., self.membrane_channel],
            markers = tile.masks[self.marker_segmentation_mask].squeeze(2),
            watershed_line = False
        )
        tile.masks[self.mask_name] = watershed[..., np.newaxis]


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
