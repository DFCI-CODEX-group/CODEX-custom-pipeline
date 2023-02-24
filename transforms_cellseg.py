import numpy as np
import skimage
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
import anndata
import itertools
import pathml
from pathml.preprocessing.transforms import Transform


class CellSegQuantifyMIF(Transform):
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
            differently for different elements. Defaults to 2.
        element_shape (str): shape of structuring element used to determine if the pixel is a boundary pixel.
            Supports "diamond", "disk", "square", and "star". Defaults to "diamond".
            See: https://scikit-image.org/docs/dev/auto_examples/numpy_operations/plot_structuring_elements.html

    References:
        Bai Y, Zhu B, Rovira-Clave X, Chen H, Markovic M, Chan CN, Su T-H, McIlwain DR, Estes JD,
        Keren L, Nolan GP and Jiang S (2021) Adjacent Cell Marker Lateral Spillover Compensation and Reinforcement
        for Multiplexed Images. Front. Immunol. 12:652631. doi: 10.3389/fimmu.2021.652631
    """
    def __init__(self, segmentation_mask=None, element_size=2, element_shape="diamond"):
        def __init__(self, flatmasks):
        self.masks = None
        self.flatmasks = segmentation_mask
        self.segmentation_mask = self.flatmask
        self.centroids = None

    def n_instances(self):
        return len(np.unique(self.flatmasks)) - 1

    def update_adjacency_value(self, adjacency_matrix, original, neighbor):
        border = False

        if original != 0 and original != neighbor:
            border = True
            if neighbor != 0:
                adjacency_matrix[int(original - 1), int(neighbor - 1)] += 1
        return border

    def update_adjacency_matrix(self, plane_mask_flattened, width, height, adjacency_matrix, index):
        mod_value_width = index % width
        origin_mask = plane_mask_flattened[index]
        left, right, up, down = False, False, False, False

        if (mod_value_width != 0):
            left = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index-1])
        if (mod_value_width != width - 1):
            right = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index+1])
        if (index >= width):
            up = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index-width])
        if (index <= len(plane_mask_flattened) - 1 - width):
            down = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index+width])
        
        if (left or right or up or down):
            adjacency_matrix[int(origin_mask - 1), int(origin_mask-1)] += 1
    def compute_channel_means_sums_compensated(image, mask):
        height, width, n_channels = image.shape
        mask_height, mask_width = mask.shape
        n_masks = len(np.unique(mask)) - 1
        channel_sums = np.zeros((n_masks, n_channels))
        channel_counts = np.zeros((n_masks, n_channels))
        if n_masks == 0:
            return channel_sums, channel_sums, channel_counts

        squashed_image = np.reshape(image, (height*width, n_channels))
        
        #masklocs = np.nonzero(self.flatmasks)
        #plane_mask = np.zeros((mask_height, mask_width), dtype = np.uint32)
        #plane_mask[masklocs[0], masklocs[1]] = masklocs[2] + 1
        #plane_mask = plane_mask.flatten()
        plane_mask = mask.flatten()
        
        adjacency_matrix = np.zeros((n_masks, n_masks))
        for i in range(len(plane_mask)):
            self.update_adjacency_matrix(plane_mask, mask_width, mask_height, adjacency_matrix, i)
            
            mask_val = plane_mask[i] - 1
            if mask_val != -1:
                channel_sums[mask_val.astype(np.int32)] += squashed_image[i]
                channel_counts[mask_val.astype(np.int32)] += 1
        
        
        # Normalize adjacency matrix
        for i in range(n_masks):
            adjacency_matrix[i] = adjacency_matrix[i] / (max(adjacency_matrix[i, i], 1) * 2)
            adjacency_matrix[i, i] = 1
        
        means = np.true_divide(channel_sums, channel_counts, out=np.zeros_like(channel_sums, dtype='float'), where=channel_counts!=0)
        results = lstsq(adjacency_matrix, means, overwrite_a=True, overwrite_b=False)
        compensated_means = np.maximum(results[0], np.zeros((1,1)))        

        return compensated_means


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

        segmentation_zero_boundary = self.get_segmentation_zero_boundary(segmentation.copy())

        counts_redsea = self.compute_redsea_counts_matrix(img=img, mask=segmentation_zero_boundary)

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
        tile.counts = self.compute_channel_means_sums_compensated(
            img = tile.image,
            segmentation = tile.masks[self.segmentation_mask].copy(),
            coords_offset = tile.coords,
        )

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
                    if 0 <= i + di <= mask.shape[0] - 1 and 0 <= j + dj <= mask.shape[0] - 1:
                        if mask[i + di, j + dj] == 0:
                            # zero pixel means that it's a boundary pixel
                            border = True
                if not border:
                    # in this case, no zero pixels were found, so it's not a border pixel
                    border_mask[i, j] = 0
        return border_mask

    @staticmethod
    def _compute_pairwise_matrix(mask):
        """
        Computes perimeters of each cell, and pairwise compensation values for each pair.
        Uses a 3x3 structuring element.
        Note that labels are counted starting at 1, but array is indexed starting at 0

        Args:
            mask (np.ndarray): (h, w) segmentation mask. Zeros are background, and pixels belonging to each of n
                cells are labelled with integers 1 to n, with a line of zeros separating adjacent regions.

        Returns:
            np.ndarray: (n_cells, n_cells) array where the i,jth element gives b_ij / P_j, i.e. the number of boundary pixels
                between cells i and j divided by the total perimeter of cell j.
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

        # loop thru zero pixels
        zero_ind_i, zero_ind_j = np.where(mask == 0)
        for i, j in zip(zero_ind_i, zero_ind_j):
            # offsets for a 3x3 structuring element centered on i,j
            adj_labels = []
            for di, dj in itertools.product([-1, 0, 1], repeat = 2):
                if 0 <= i + di <= mask.shape[0] - 1 and 0 <= j + dj <= mask.shape[0] - 1:
                    label = mask[i + di, j + dj]
                    if label != 0 and label not in adj_labels:
                        adj_labels.append(label)

            # increment perimeter counts
            # need to subtract 1 because labels start at 1 but index starts at 0
            for label in adj_labels:
                cell_perimeters[label - 1] += 1

            # increment shared perimeter counts
            # need to subtract 1 because labels start at 1 but index starts at 0
            for label1, label2 in itertools.permutations(adj_labels, 2):
                cell_adjacencies[label1 - 1, label2 - 1] += 1

        # raise if any of the cell perimeters are zero, since this will introduce NaNs in the next division step
        # also this really shouldn't ever happen, since every region has to have a perimeter>0
        if np.any(cell_perimeters == 0):
            raise ValueError("Cell perimeters in _compute_pairwise_matrix() contains zeros!!")

        # divide to get fraction
        cell_adjacencies = cell_adjacencies / cell_perimeters

        return cell_adjacencies

    def compute_redsea_counts_matrix(self, img, mask):
        """
        Computes counts matrix using REDSEA algorithm

        Args:
            img (np.ndarray): (h, w, n_channels) Input image
            mask (np.ndarray): (h, w) segmentation mask. Zeros are background, and pixels belonging to each of n cells
                are labelled with integers, with a line of zeros separating adjacent regions.
                Labels will be relabeled from 1 to n.

        Returns:
            np.ndarray: counts matrix (n_cells, n_channels)
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
        mask_border = self._mask_to_border_mask(
            mask,
            structuring_element = self.structuring_element,
            element_size = self.element_size
        )

        # get cell-cell interaction matrix
        cell_pair_weights = self._compute_pairwise_matrix(mask)

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
        counts_subtract = cell_pair_weights @ counts_border

        # now apply reinforcement and subtraction
        counts_redsea = counts + counts_border - counts_subtract

        # normalize by cell area
        counts_redsea = np.diag([1 / cell_size for cell_size in cell_sizes]) @ counts_redsea

        # clip negative values to 0
        counts_redsea = counts_redsea.clip(0)

        return counts_redsea

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
