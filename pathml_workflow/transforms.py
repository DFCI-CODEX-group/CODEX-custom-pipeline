import numpy as np
import skimage
from skimage.measure import regionprops_table
import anndata
import itertools
import pathml
from pathml.preprocessing.transforms import Transform


class REDSEAQuantifyMIF(Transform):
    """
    Convert segmented image into anndata.AnnData counts object `AnnData <https://anndata.readthedocs.io/en/latest/>`_,
    using the REinforcement Dynamic Spillover EliminAtion (REDSEA) algorithm for spillover correction.
    For each cell, signal from the boundary pixels is reinforced, while signal from boundary pixels of neighboring
    cells is subtracted.

    Should be used with a segmentation mask where zeros are background, and pixels belonging to each of n
    cells are labelled with integers 1 to n, with a line of zeros separating adjacent regions.
    For example, the output of ``skimage.segmentation.watershed`` with ``watershed_line=True``.

    Counts objects are used to interface with the Python single cell analysis ecosystem in
    `Scanpy <https://scanpy.readthedocs.io/en/stable/>`_.
    The counts object contains a summary of channel statistics in each cell along with its coordinate.

    Some parts of this implementation are based on the original implementation by the authors:
    https://github.com/nolanlab/REDSEA/tree/master/code/REDSEApy

    Args:
        segmentation_mask (str): key indicating which mask to use as label image
        element_size (int): width of structuring element used to determine if the pixel is a boundary pixel.
            Defaults to 2.
        element_shape (str): shape of structuring element used to determine if the pixel is a boundary pixel.
            Defaults to "diamond" to use ``skimage.morphology.diamond``

    References:
        Bai Y, Zhu B, Rovira-Clave X, Chen H, Markovic M, Chan CN, Su T-H, McIlwain DR, Estes JD,
        Keren L, Nolan GP and Jiang S (2021) Adjacent Cell Marker Lateral Spillover Compensation and Reinforcement
        for Multiplexed Images. Front. Immunol. 12:652631. doi: 10.3389/fimmu.2021.652631
    """
    def __init__(self, segmentation_mask=None, element_size=2, element_shape="diamond"):
        if element_shape == "diamond":
            self.structuring_element = skimage.morphology.diamond(element_size)
        else:
            raise ValueError(f"input element shape {element_shape} not valid")

        self.element_size = element_size
        self.element_shape = element_shape
        self.segmentation_mask = segmentation_mask

    def F(self, img, segmentation, coords_offset=(0, 0)):
        """
        Functional implementation of marker quantification using the REDSEA algorithm

        Args:
            img (np.ndarray): (h, w, n_channels) Input image
            segmentation (np.ndarray): (h, w) segmentation mask. Zeros are background, and pixels belonging to each of n
                cells are labelled with integers 1 to n, with a line of zeros separating adjacent regions.
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

        counts_redsea = self.compute_redsea_counts_matrix(img=img, mask=segmentation)

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
            segmentation = tile.masks[self.segmentation_mask],
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
                cells are labelled with integers 1 to n, with a line of zeros separating adjacent regions.
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

        # divide to get fraction
        cell_adjacencies = cell_adjacencies / cell_perimeters

        return cell_adjacencies

    def compute_redsea_counts_matrix(self, img, mask):
        """
        Computes counts matrix using REDSEA algorithm

        Args:
            img (np.ndarray): (h, w, n_channels) Input image
            mask (np.ndarray): (h, w) segmentation mask. Zeros are background, and pixels belonging to each of n
                cells are labelled with integers 1 to n, with a line of zeros separating adjacent regions.

        Returns:
            np.ndarray: counts matrix (n_cells, n_channels)
        """
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
            watershed_line = self.watershed_line
        )
        tile.masks[self.mask_name] = watershed[..., np.newaxis]
