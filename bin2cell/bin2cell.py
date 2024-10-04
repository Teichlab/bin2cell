from __future__ import annotations

from scanpy import read_10x_h5
from scanpy import logging as logg

import json
from pathlib import Path, PurePath
from typing import BinaryIO, Literal
import pandas as pd
from matplotlib.image import imread

#actual bin2cell dependencies start here
#the ones above are for read_visium()
from stardist.plot import render_label
from copy import deepcopy
import tifffile as tf
import scipy.spatial
import scipy.sparse
import scipy.stats
import anndata as ad
import skimage
import scanpy as sc
import numpy as np
import os

from PIL import Image
#setting needed so PIL can load the large TIFFs
Image.MAX_IMAGE_PIXELS = None

#setting needed so cv2 can load the large TIFFs
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
import cv2

#NOTE ON DIMENSIONS WITHIN ANNDATA AND BEYOND
#.obs["array_row"] matches .obsm["spatial"][:,1] matches np.array image [:,0]
#.obs["array_col"] matches .obsm["spatial"][:,0] matches np.array image [:,1]
#array coords start relative to some magical point on the grid (seen bottom left/top right)
#and can require flipping the array row or column to match the image orientation
#row/col do seem to consistenly refer to the stated np.array image dimensions though
#spatial starts in top left corner of image, matching what np.array image is doing
#of note, cv2 treats [:,1] as dim 0 and [:,0] as dim 1, despite working on np.arrays
#also cv2 works with channels in a BGR order, while everything else is RGB

def load_image(image_path, gray=False, dtype=np.uint8):
    '''
    Efficiently load an external image and return it as an RGB numpy array.
    
    Input
    -----
    image_path : ``filepath``
        Path to image to be loaded.
    gray : ``bool``, optional (default: ``False``)
        Whether to turn image to grayscale before returning
    dtype : ``numpy.dtype``, optional (default: ``numpy.uint8``)
        Data type of the numpy array output.
    '''
    img = cv2.imread(image_path)
    #loaded as BGR by default, turn to the expected RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #optionally turn to greyscale
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #copy=False means that there's no extra copy made if the dtype already matches
    #which it will for np.uint8
    return img.astype(dtype, copy=False)

def normalize(img):
    '''
    Extremely naive reimplementation of default ``cbsdeep.utils.normalize()`` 
    percentile normalisation, with a lower RAM footprint than the original.
    
    Input
    -----
    img : ``numpy.array``
        Numpy array image to normalise
    '''
    eps = 1e-20
    mi = np.percentile(img,3)
    ma = np.percentile(img,99.8)
    return ((img - mi) / ( ma - mi + eps ))

def stardist(image_path, labels_npz_path, stardist_model="2D_versatile_he", block_size=4096, min_overlap=128, context=128, **kwargs):
    '''
    Segment an image with StarDist. Supports both the fluorescence and 
    H&E models. The identified object labels will be converted to a 
    sparse matrix and written to drive in ``.npz``.
    
    Input
    -----
    image_path : ``filepath``
        Path to image to be segmented.
    labels_npz_path : ``filepath``
        Path to write object labels output. Can be easily loaded via 
        ``scipy.sparse.load_npz()``.
    stardist_model : ``str``, optional (default: ``"2D_versatile_he"``)
        Use ``"2D_versatile_he"`` for segmenting H&E images or 
        ``"2D_versatile_fluo"`` for segmenting single-channel 
        images (GEX-derived or IF)
    block_size : ``int``, optional (default: 4096)
        StarDist ``predict_instances_big()`` input. Length of square edge 
        of the image to process as a single tile. 
    min_overlap : ``int``, optional (default: 128)
        StarDist ``predict_instances_big()`` input. Minimum overlap between 
        adjacent tiles, in each dimension.
    context : ``int``, optional (default: 128)
        StarDist ``predict_instances_big()`` input. Amount of image context 
        on all sides of a block, which is discarded.
    kwargs
        Any additional arguments to pass to StarDist. Practically most likely 
        to be ``prob_thresh`` for controlling the stringency of calling 
        objects.
    '''
    #using stardist models requires tensorflow, avoid global import
    from stardist.models import StarDist2D
    #load and percentile normalize image, following stardist demo protocol
    #turn it to np.float16 pre normalisation to keep RAM footprint minimal
    img = load_image(image_path, gray=(stardist_model=="2D_versatile_fluo"), dtype=np.float16)
    img = normalize(img)
    #use pretrained stardist model
    model = StarDist2D.from_pretrained(stardist_model)
    #will need to specify axes shortly, which are model dependent
    if stardist_model == "2D_versatile_he":
        #3D image, got the axes YXC from the H&E model config.json
        model_axes = "YXC"
    elif stardist_model == "2D_versatile_fluo":
        #2D image, got the axes YX from logic and trying and it working
        model_axes = "YX"
    #run predict_instances_big() to perform automated tiling of the input
    #this is less parameterised than predict_instances, needed to pass axes too
    #pass any other **kwargs to the thing, passing them on internally
    #in practice this is going to be prob_thresh
    labels, _ = model.predict_instances_big(img, axes=model_axes, 
                                            block_size=block_size, 
                                            min_overlap=min_overlap, 
                                            context=context, 
                                            **kwargs
                                           )
    #store resulting labels as sparse matrix NPZ - super efficient space wise
    labels_sparse = scipy.sparse.csr_matrix(labels)
    scipy.sparse.save_npz(labels_npz_path, labels_sparse)
    print("Found "+str(len(np.unique(labels_sparse.data)))+" objects")

def view_stardist_labels(image_path, labels_npz_path, crop, **kwargs):
    '''
    Use StarDist's label rendering to view segmentation results in a crop 
    of the input image.
    
    Input
    -----
    image_path : ``filepath``
        Path to image that was segmented.
    labels_npz_path : ``filepath``
        Path to sparse labels generated by ``b2c.stardist()``.
    crop : tuple of ``int``
        A PIL-formatted crop specification - a four integer tuple, 
        provided as (left, upper, right, lower) coordinates.
    kwargs
        Any additional arguments to pass to StarDist's ``render_labels()``. 
        Practically most likely to be ``normalize_img``.
    '''
    #PIL is better at handling crops memory efficiently than cv2
    img = Image.open(image_path)
    img = img.crop(crop)
    #stardist likes images on a 0-1 scale
    img = np.array(img)/255
    #load labels and subset to area of interest
    #crop is (left, upper, right, lower)
    #https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #upper:lower, left:right
    labels_sparse = labels_sparse[crop[1]:crop[3], crop[0]:crop[2]]
    #reset labels within crop to start from 1 and go up by 1
    #which makes the stardist view more colourful in objects
    #calling this on [5,7,7,9] yields [1,2,2,3] which is what we want
    labels_sparse.data = scipy.stats.rankdata(labels_sparse.data, method="dense")
    labels = np.array(labels_sparse.todense())
    return render_label(labels, img=img, **kwargs)

#as PR'd to scanpy: https://github.com/scverse/scanpy/pull/2992
def read_visium(
    path: Path | str,
    genome: str | None = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool | None = True,
    source_image_path: Path | str | None = None,
    spaceranger_image_path: Path | str | None = None,
) -> AnnData:
    """\
    Read 10x-Genomics-formatted visum dataset.

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.

    See :func:`~scanpy.pl.spatial` for a compatible plotting function.

    .. _Space Ranger output docs: https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    source_image_path
        Path to the high-resolution tissue image. Path will be included in
        `.uns["spatial"][library_id]["metadata"]["source_image_path"]`.
    spaceranger_image_path
        Path to the folder containing the spaceranger output hires/lowres tissue images. If `None`, 
        will go with the `spatial` folder of the provided `path`.

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:

    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names for a feature barcode matrix, probe names for a probe bc matrix
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.obs`\\ `[filtered_barcodes]`
        filtered barcodes if present in the matrix
    :attr:`~anndata.AnnData.var`
        Any additional metadata present in /matrix/features is read in.
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version', 'source_image_path'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    path = Path(path)
    #if not provided, assume the hires/lowres images are in the same folder as everything
    #except in the spatial subdirectory
    if spaceranger_image_path is None:
        spaceranger_image_path = path / "spatial"
    else:
        spaceranger_image_path = Path(spaceranger_image_path)
    adata = read_10x_h5(path / count_file, genome=genome)

    adata.uns["spatial"] = dict()

    from h5py import File

    with File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)
    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        tissue_positions_file = (
            path / "spatial/tissue_positions.csv"
            if (path / "spatial/tissue_positions.csv").exists()
            else path / "spatial/tissue_positions.parquet" if (path / "spatial/tissue_positions.parquet").exists()
            else path / "spatial/tissue_positions_list.csv"
        )
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            hires_image=spaceranger_image_path / "tissue_hires_image.png",
            lowres_image=spaceranger_image_path / "tissue_lowres_image.png",
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logg.warning(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            try:
                adata.uns["spatial"][library_id]["images"][res] = imread(
                    str(files[f"{res}_image"])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        # read coordinates
        if files["tissue_positions_file"].name.endswith(".csv"):
            positions = pd.read_csv(
                files["tissue_positions_file"],
                header=0 if tissue_positions_file.name == "tissue_positions.csv" else None,
                index_col=0,
            )
        elif files["tissue_positions_file"].name.endswith(".parquet"):
            positions = pd.read_parquet(files["tissue_positions_file"])
            #need to set the barcode to be the index
            positions.set_index("barcode", inplace=True)
        positions.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm["spatial"] = adata.obs[
            ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        adata.obs.drop(
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata

def destripe_counts(adata, counts_key="n_counts", adjusted_counts_key="n_counts_adjusted"):
    '''
    Scale each row (bin) of ``adata.X`` to have ``adjusted_counts_key`` 
    rather than ``counts_key`` total counts.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` 
        and ``adjusted_counts_key`` in ``.obs``.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column storing the desired destriped counts per bin.
    '''
    #scanpy's utility function to make sure the anndata is not a view
    #if it is a view then weird stuff happens when you try to write to its .X
    sc._utils.view_to_actual(adata)
    #adjust the count matrix to have n_counts_adjusted sum per bin (row)
    #premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    bin_scaling = scipy.sparse.diags(adata.obs[adjusted_counts_key]/adata.obs[counts_key])
    adata.X = bin_scaling.dot(adata.X)

def destripe(adata, quantile=0.99, counts_key="n_counts", factor_key="destripe_factor", adjusted_counts_key="n_counts_adjusted", adjust_counts=True):
    '''
    Correct the raw counts of the input object for known variable width of 
    VisiumHD 2um bins. Scales the total UMIs per bin on a per-row and 
    per-column basis, dividing by the specified ``quantile``. The resulting 
    value is stored in ``.obs[factor_key]``, and is multiplied by the 
    corresponding total UMI ``quantile`` to get ``.obs[adjusted_counts_key]``.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` in 
        ``.obs``.
    quantile : ``float``, optional (default: 0.99)
        Which row/column quantile to use for the computation.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    factor_key : ``str``, optional (default: ``"destripe_factor"``)
        Name of ``.obs`` column to hold computed factor prior to reversing to 
        count space.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column for storing the destriped counts per bin.
    adjust_counts : ``bool``, optional (default: ``True``)
        Whether to use the computed adjusted count total to adjust the counts in 
        ``adata.X``.
    '''
    #apply destriping via sequential quantile scaling
    #get specified quantile per row
    quant = adata.obs.groupby("array_row")[counts_key].quantile(quantile)
    #divide each row by its quantile (order of obs[counts_key] and obs[array_row] match)
    adata.obs[factor_key] = adata.obs[counts_key] / adata.obs["array_row"].map(quant)
    #repeat on columns
    quant = adata.obs.groupby("array_col")[factor_key].quantile(quantile)
    adata.obs[factor_key] /= adata.obs["array_col"].map(quant)
    #propose adjusted counts as the global quantile multipled by the destripe factor
    adata.obs[adjusted_counts_key] = adata.obs[factor_key] * np.quantile(adata.obs[counts_key], quantile)
    #correct the count space unless told not to
    if adjust_counts:
        destripe_counts(adata, counts_key=counts_key, adjusted_counts_key=adjusted_counts_key)

def check_array_coordinates(adata, row_max=3349, col_max=3349):
    '''
    Assess the relationship between ``.obs["array_row"]``/``.obs["array_col"]`` 
    and ``.obsm["spatial"]``, as the array coordinates tend to have their 
    origin in places that isn't the top left of the image. Array coordinates 
    are deemed to be flipped or not based on the Pearson correlation with the 
    corresponding spatial dimension. Creates ``.uns["bin2cell"]["array_check"]`` 
    that is used by ``b2c.grid_image()``, ``b2c.insert_labels()`` and 
    ``b2c.get_crop()``.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    row_max : ``int``, optional (default: 3349)
        Maximum possible ``array_row`` value.
    col_max : ``int``, optional (default: 3349)
        Maximum possible ``array_col`` value.
    '''
    #store the calls here
    if not "bin2cell" in adata.uns:
        adata.uns["bin2cell"] = {}
    adata.uns["bin2cell"]["array_check"] = {}
    #we'll need to check both the rows and columns
    for axis in ["row", "col"]:
        #we may as well store the maximum immediately
        adata.uns["bin2cell"]["array_check"][axis] = {}
        if axis == "row":
            adata.uns["bin2cell"]["array_check"][axis]["max"] = row_max
        elif axis == "col":
            adata.uns["bin2cell"]["array_check"][axis]["max"] = col_max
        #are we going to be extracting values for a single col or row?
        #set up where we'll be looking to get values to correlate
        if axis == "col":
            single_axis = "row"
            #spatial[:,0] matches axis_col (note at start)
            spatial_axis = 0
        elif axis == "row":
            single_axis = "col"
            #spatial[:,1] matches axis_row (note at start)
            spatial_axis = 1
        #get the value of the other axis with the highest number of bins present
        val = adata.obs['array_'+single_axis].value_counts().index[0]
        #get a boolean mask of the bins of that value
        mask = (adata.obs['array_'+single_axis] == val)
        #use the mask to get the spatial and array coordinates to compare
        array_vals = adata.obs.loc[mask,'array_'+axis].values
        spatial_vals = adata.obsm['spatial'][mask, spatial_axis]
        #check whether they're positively or negatively correlated
        if scipy.stats.pearsonr(array_vals, spatial_vals)[0] < 0:
            adata.uns["bin2cell"]["array_check"][axis]["flipped"] = True
        else:
            adata.uns["bin2cell"]["array_check"][axis]["flipped"] = False

def grid_image(adata, val, log1p=False, mpp=2, sigma=None, save_path=None):
    '''
    Create an image of a specified ``val`` across the array coordinate grid. 
    Orientation matches the morphology image and spatial coordinates.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Should have array coordinates evaluated by 
        calling ``b2c.check_array_coordinates()``.
    val : ``str``
        ``.obs`` column or variable name to visualise.
    log1p : ``bool``, optional (default: ``False``)
        Whether to log1p-transform the values in the image.
    mpp : ``float``, optional (default: 2)
        Microns per pixel of the output image.
    sigma : ``float`` or ``None``, optional (default: ``None``)
        If not ``None``, will run the final image through 
        ``skimage.filters.gaussian()`` with the provided sigma value.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for 
        StarDist use). If not provided, will return image.
    '''
    #pull out the values for the image. start by checking .obs
    if val in adata.obs.columns:
        vals = adata.obs[val].values.copy()
    elif val in adata.var_names:
        #if not in obs, it's presumably in the feature space
        vals = adata[:, val].X
        #may be sparse
        if scipy.sparse.issparse(vals):
            vals = vals.todense()
        #turn it to a flattened numpy array so it plays nice
        vals = np.asarray(vals).flatten()
    else:
        #failed to find
        raise ValueError('"'+val+'" not located in ``.obs`` or ``.var_names``')
    #make the values span from 0 to 255
    vals = (255 * (vals-np.min(vals))/(np.max(vals)-np.min(vals))).astype(np.uint8)
    #optionally log1p
    if log1p:
        vals = np.log1p(vals)
        vals = (255 * (vals-np.min(vals))/(np.max(vals)-np.min(vals))).astype(np.uint8)
    #spatial coordinates match what's going on in the image, array coordinates may not
    #have we checked if the array row/col need flipping?
    if not "bin2cell" in adata.uns:
        check_array_coordinates(adata)
    elif not "array_check" in adata.uns["bin2cell"]:
        check_array_coordinates(adata)
    #can now create an empty image the shape of the grid and stick the values in based on the coordinates
    #need to nudge up the dimensions by 1 as python is zero-indexed
    img = np.zeros((adata.uns["bin2cell"]["array_check"]["row"]["max"]+1, 
                    adata.uns["bin2cell"]["array_check"]["col"]["max"]+1), 
                   dtype=np.uint8)
    img[adata.obs['array_row'], adata.obs['array_col']] = vals
    #check if the row or column need flipping
    if adata.uns["bin2cell"]["array_check"]["row"]["flipped"]:
        img = np.flip(img, axis=0)
    if adata.uns["bin2cell"]["array_check"]["col"]["flipped"]:
        img = np.flip(img, axis=1)
    #resize image to appropriate mpp. bins are 2um apart, so current mpp is 2
    #need to reverse dimensions relative to the array for cv2, and turn to int
    if mpp != 2:
        dim = np.round(np.array(img.shape) * 2/mpp).astype(int)[::-1]
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    #run through the gaussian filter if need be
    if sigma is not None:
        img = skimage.filters.gaussian(img, sigma=sigma)
        img = (255 * (img-np.min(img))/(np.max(img)-np.min(img))).astype(np.uint8)
    #save or return image
    if save_path is not None:
        cv2.imwrite(save_path, img)
    else:
        return img

def check_bin_image_overlap(adata, img, overlap_threshold=0.9):
    '''
    Assess the number of bins that fall within the source image coordinate 
    space. If an insufficient proportion are captured then throw an informative 
    error.
    
    Input
    -----
    adata : ``AnnData``
        2um bin Visium object.
    img : ``np.array``
        Loaded full resolution morphology image, prior to any cropping/scaling.
    overlap_threshold : ``float``, optional (default: 0.9)
        Throw the error if fewer than this fraction of bin spatial coordinates 
        fall within the dimensions of the image.
    '''
    #spatial[:,1] matches img[:,0] and spatial[:,0] matches img[:,1]
    #check how many fall within the dimensions, and get a fraction of total bin count
    overlap = np.sum((adata.obsm["spatial"][:,1] < img.shape[0]) & (adata.obsm["spatial"][:,0] < img.shape[1])) / adata.shape[0]
    if overlap < overlap_threshold:
        #something is amiss. print a bunch of diagnostics
        print("Source image dimensions: "+ str(img.shape))
        #the end user does not need to know about the messiness of the representations
        #pre-format the spatial maxima to match the dimensions of the image
        print("Corresponding ``.obsm['spatial']`` maxima: "+str(np.max(adata.obsm["spatial"], axis=0)[::-1]))
        raise ValueError("Only "+str(100*overlap)+"% of bins fall within image. Are you running with ``source_image_path`` set to the full resolution morphology image, as used for ``--image`` in Spaceranger?")

def mpp_to_scalef(adata, mpp):
    '''
    Compute a scale factor for a specified mpp value.
    
    Input
    -----
    adata : ``AnnData``
        2um bin Visium object.
    mpp : ``float``
        Microns per pixel to report scale factor for.
    '''
    #identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    #get original image mpp value
    mpp_source = adata.uns['spatial'][library]['scalefactors']['microns_per_pixel']
    #our scale factor is the original mpp divided by the new mpp
    return mpp_source/mpp

def get_mpp_coords(adata, basis="spatial", spatial_key="spatial", mpp=None):
    '''
    Get an mpp-adjusted representation of spatial or array coordinates of the 
    provided object. Origin in top left, dimensions correspond to ``np.array()`` 
    representation of image (``[:,0]`` is up-down, ``[:,1]`` is left-right). 
    The resulting coordinates are integers for ease of retrieval of labels from 
    arrays or defining crops.
    
    adata : ``AnnData``
        2um bin VisiumHD object.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether to get ``"spatial"`` or ``"array"`` coordinates. The former is 
        the source morphology image, the latter is a GEX-based grid representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``. 
        Rounded coordinates will be used to represent each bin when retrieving 
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The mpp value. Mandatory for GEX (``basis="array"``), if not provided 
        with morphology (``basis="spatial"``) will assume full scale image.
    '''
    #if we're using array coordinates, is there an mpp provided?
    if basis == "array" and mpp is None:
        raise ValueError("Need to specify mpp if working with array coordinates.")
    if basis == "spatial":
        if mpp is not None:
            #get necessary scale factor
            scalef = mpp_to_scalef(adata, mpp=mpp)
        else:
            #no mpp implies full blown morphology image, so scalef is 1
            scalef = 1
        #get the matching coordinates, rounding to integers makes this agree
        #need to reverse them here to make the coordinates match the image, as per note at start
        #multiply by the scale factor to account for possible custom mpp morphology image
        coords = (adata.obsm[spatial_key]*scalef).astype(int)[:,::-1]
    elif basis == "array":
        #generate the pixels in the GEX image at the specified mpp
        #which actually correspond to the locations of the bins
        #easy to define scale factor as starting array mpp is 2
        scalef = 2/mpp
        coords = np.round(adata.obs[['array_row','array_col']].values*scalef).astype(int)
        #need to flip axes maybe
        #need to scale up maximum appropriately
        if adata.uns["bin2cell"]["array_check"]["row"]["flipped"]:
            coords[:,0] = np.round(adata.uns["bin2cell"]["array_check"]["row"]["max"]*scalef).astype(int) - coords[:,0]
        if adata.uns["bin2cell"]["array_check"]["col"]["flipped"]:
            coords[:,1] = np.round(adata.uns["bin2cell"]["array_check"]["col"]["max"]*scalef).astype(int) - coords[:,1]
    return coords

def get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=0):
    '''
    Get a PIL-formatted crop tuple from a provided object and coordinate 
    representation.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether to use ``"spatial"`` or ``"array"`` coordinates. The former is 
        the source morphology image, the latter is a GEX-based grid representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``. 
        Rounded coordinates will be used to represent each bin when retrieving 
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The micron per pixel value to use. Mandatory for GEX (``basis="array"``), 
        if not provided with morphology (``basis="spatial"``) will assume full scale 
        image.
    buffer : ``int``, optional (default: 0)
        How many extra pixels to include to each side the cropped grid for 
        extra visualisation.
    '''
    #get the appropriate coordinates, be they spatial or array, at appropriate mpp
    coords = get_mpp_coords(adata, basis=basis, spatial_key=spatial_key, mpp=mpp)
    #PIL crop is defined as a tuple of (left, upper, right, lower) coordinates
    #coords[:,0] is up-down, coords[:,1] is left-right
    #don't forget to add/remove buffer, and to not go past 0
    return (np.max([np.min(coords[:,1])-buffer, 0]), 
            np.max([np.min(coords[:,0])-buffer, 0]), 
            np.max(coords[:,1])+buffer, 
            np.max(coords[:,0])+buffer
           )

def scaled_he_image(adata, mpp=1, crop=True, buffer=150, spatial_cropped_key="spatial_cropped", save_path=None):
    '''
    Create a custom microns per pixel render of the full scale H&E image for 
    visualisation and downstream application. Store resulting image and its 
    corresponding size factor in the object. If cropping to just the spatial 
    grid, also store the cropped spatial coordinates. Optionally save to file.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Path to high resolution H&E image provided via 
        ``source_image_path`` to ``b2c.read_visium()``.
    mpp : ``float``, optional (default: 1)
        Microns per pixel of the desired H&E image to create.
    crop : ``bool``, optional (default: ``True``)
        If ``True``, will limit the image to the actual spatial coordinate area, 
        with ``buffer`` added to each dimension.
    buffer : ``int``, optional (default: 150)
        Only used with ``crop=True``. How many extra pixels (in original 
        resolution) to include on each side of the captured spatial grid.
    spatial_cropped_key : ``str``, optional (default: ``"spatial_cropped"``)
        Only used with ``crop=True``. ``.obsm`` key to store the adjusted 
        spatial coordinates in.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for 
        StarDist use).
    '''
    #identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    #retrieve specified source image path and load it
    img = load_image(adata.uns['spatial'][library]['metadata']['source_image_path'])
    #assess that the image actually matches the spatial coordinates
    #if not, inform the user what image they should retrieve and use
    check_bin_image_overlap(adata, img)
    #crop image if necessary
    if crop:
        crop_coords = get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer)
        #this is already capped at a minimum of 0, so can just subset freely
        #left, upper, right, lower; image is up-down, left-right
        img = img[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2], :]
        #need to move spatial so it starts at the new crop top left point
        #spatial[:,1] is up-down, spatial[:,0] is left-right
        adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
        adata.obsm[spatial_cropped_key][:,0] -= crop_coords[0]
        adata.obsm[spatial_cropped_key][:,1] -= crop_coords[1]
    #reshape image to desired microns per pixel
    #get necessary scale factor for the custom mpp
    #multiply dimensions by this to get the shrunken image size
    #multiply .obsm['spatial'] by this to get coordinates matching the image
    scalef = mpp_to_scalef(adata, mpp=mpp)
    #need to reverse dimension order and turn to int for cv2
    dim = (np.array(img.shape[:2])*scalef).astype(int)[::-1]
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #we have everything we need. store in object
    adata.uns['spatial'][library]['images'][str(mpp)+"_mpp"] = img
    #the scale factor needs to be prefaced with "tissue_"
    adata.uns['spatial'][library]['scalefactors']['tissue_'+str(mpp)+"_mpp_scalef"] = scalef
    if save_path is not None:
        #cv2 expects BGR channel order, we're working with RGB
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def scaled_if_image(adata, channel, mpp=1, crop=True, buffer=150, spatial_cropped_key="spatial_cropped", save_path=None):
    '''
    Create a custom microns per pixel render of the full scale IF image for 
    visualisation and downstream application. Store resulting image and its 
    corresponding size factor in the object. If cropping to just the spatial 
    grid, also store the cropped spatial coordinates. Optionally save to file.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Path to high resolution IF image provided via 
        ``source_image_path`` to ``b2c.read_visium()``.
    channel : ``int``
        The channel of the IF image holding the DAPI capture.
    mpp : ``float``, optional (default: 1)
        Microns per pixel of the desired IF image to create.
    crop : ``bool``, optional (default: ``True``)
        If ``True``, will limit the image to the actual spatial coordinate area, 
        with ``buffer`` added to each dimension.
    buffer : ``int``, optional (default: 150)
        Only used with ``crop=True``. How many extra pixels (in original 
        resolution) to include on each side of the captured spatial grid.
    spatial_cropped_key : ``str``, optional (default: ``"spatial_cropped"``)
        Only used with ``crop=True``. ``.obsm`` key to store the adjusted 
        spatial coordinates in.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for 
        StarDist use).
    '''
    #identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    #pull out specified channel from IF tiff via tifffile
    #pretype to float32 for space while working with plots (float16 does not)
    img = tf.imread(adata.uns['spatial'][library]['metadata']['source_image_path'], key=channel).astype(np.float32)
    #assess that the image actually matches the spatial coordinates
    #if not, inform the user what image they should retrieve and use
    check_bin_image_overlap(adata, img)
    #this can be dark, apply stardist normalisation to fix
    img = normalize(img)
    #actually cap the values - currently there are sub 0 and above 1 entries
    img[img<0] = 0
    img[img>1] = 1
    #crop image if necessary
    if crop:
        crop_coords = get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer)
        #this is already capped at a minimum of 0, so can just subset freely
        #left, upper, right, lower; image is up-down, left-right
        img = img[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
        #need to move spatial so it starts at the new crop top left point
        #spatial[:,1] is up-down, spatial[:,0] is left-right
        adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
        adata.obsm[spatial_cropped_key][:,0] -= crop_coords[0]
        adata.obsm[spatial_cropped_key][:,1] -= crop_coords[1]
    #reshape image to desired microns per pixel
    #get necessary scale factor for the custom mpp
    #multiply dimensions by this to get the shrunken image size
    #multiply .obsm['spatial'] by this to get coordinates matching the image
    scalef = mpp_to_scalef(adata, mpp=mpp)
    #need to reverse dimension order and turn to int for cv2
    dim = (np.array(img.shape[:2])*scalef).astype(int)[::-1]
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #we have everything we need. store in object
    adata.uns['spatial'][library]['images'][str(mpp)+"_mpp"] = img
    #the scale factor needs to be prefaced with "tissue_"
    adata.uns['spatial'][library]['scalefactors']['tissue_'+str(mpp)+"_mpp_scalef"] = scalef
    if save_path is not None:
        #cv2 expects BGR channel order, we have a greyscale image
        #oh also we should make it a uint8 as otherwise stuff won't work
        cv2.imwrite(save_path, cv2.cvtColor((255*img).astype(np.uint8), cv2.COLOR_GRAY2BGR))

def insert_labels(adata, labels_npz_path, basis="spatial", spatial_key="spatial", mpp=None, labels_key="labels"):
    '''
    Load StarDist segmentation results and store them in the object. Labels 
    will be stored as integers, with 0 being unassigned to an object.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    labels_npz_path : ``filepath``
        Path to sparse labels generated by ``b2c.stardist()``.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether the image represents ``"spatial"`` or ``"array"`` coordinates. 
        The former is the source morphology image, the latter is a GEX-based grid 
        representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``. 
        Rounded coordinates will be used to represent each bin when retrieving 
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The mpp value that was used to generate the segmented image. Mandatory 
        for GEX (``basis="array"``), if not provided with morphology 
        (``basis="spatial"``) will assume full scale image.
    labels_key : ``str``, optional (default: ``"labels"``)
        ``.obs`` key to store the labels under.
    '''
    #load sparse segmentation results
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #may as well stash that path in .uns['bin2cell'] since we have it
    if "bin2cell" not in adata.uns:
        adata.uns["bin2cell"] = {}
    if "labels_npz_paths" not in adata.uns["bin2cell"]:
        adata.uns["bin2cell"]["labels_npz_paths"] = {}
    #store as absolute path if it's relative
    if labels_npz_path[0] != "/":
        npz_prefix = os.getcwd()+"/"
    else:
        npz_prefix = ""
    adata.uns["bin2cell"]["labels_npz_paths"][labels_key] = npz_prefix + labels_npz_path
    #get the appropriate coordinates, be they spatial or array, at appropriate mpp
    coords = get_mpp_coords(adata, basis=basis, spatial_key=spatial_key, mpp=mpp)
    #there is a possibility that some coordinates will fall outside labels_sparse
    #start by pregenerating an obs column of all zeroes so all bins are covered
    adata.obs[labels_key] = 0
    #can now construct a mask defining which coordinates fall within range
    #apply the mask to the coords and the obs to just go for the relevant bins
    mask = ((coords[:,0] >= 0) & 
            (coords[:,0] < labels_sparse.shape[0]) & 
            (coords[:,1] >= 0) & 
            (coords[:,1] < labels_sparse.shape[1])
           )
    #pull out the cell labels for the coordinates, can just index the sparse matrix with them
    #insert into bin object, need to turn it into a 1d numpy array from a 1d numpy matrix first
    adata.obs.loc[mask, labels_key] = np.asarray(labels_sparse[coords[mask,0], coords[mask,1]]).flatten()

def expand_labels(adata, labels_key="labels", expanded_labels_key="labels_expanded", max_bin_distance=2, volume_ratio=4, k=4, subset_pca=True):
    '''
    Expand StarDist segmentation results to bins a maximum distance away in 
    the array coordinates. In the event of multiple equidistant bins with 
    different labels, ties are broken by choosing the closest bin in a PCA 
    representation of gene expression. The resulting labels will be integers, 
    with 0 being unassigned to an object.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts.
    labels_key : ``str``, optional (default: ``"labels"``)
        ``.obs`` key holding the labels to be expanded. Integers, with 0 being 
        unassigned to an object.
    expanded_labels_key : ``str``, optional (default: ``"labels_expanded"``)
        ``.obs`` key to store the expanded labels under.
    max_bin_distance : ``int`` or ``None``, optional (default: 2)
        Maximum number of bins to expand the nuclear labels by. Specifying 
        ``None`` will use the per-label expansion detailed below.
    volume_ratio : ``float``, optional (default: 4)
        If ``max_bin_distance = None``, a per-label expansion will be proposed 
        as ``ceil((volume_ratio**(1/3)-1) * sqrt(n_bins/pi))``, where 
        ``n_bins`` is the number of bins for the corresponding pre-expansion 
        label. Default based on cell line 
        `data <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8893647/>`_
    k : ``int``, optional (default: 4)
        Number of assigned spatial coordinate bins to find as potential nearest 
        neighbours for each unassigned bin.
    subset_pca : ``bool``, optional (default: ``True``)
        If ``True``, will obtain the PCA representation of just the bins 
        involved in the tie breaks rather than the full bin space. Results in 
        a slightly different embedding at a lower resource footprint.
    '''
    #this is where the labels will go
    adata.obs[expanded_labels_key] = adata.obs[labels_key].values.copy()
    #get out our array grid, and preexisting labels
    coords = adata.obs[["array_row","array_col"]].values
    labels = adata.obs[labels_key].values
    #we'll be splitting the space in two - the bins with labels, and those without
    object_mask = (labels != 0)
    #get their indices in cell space
    full_reference_inds = np.arange(adata.shape[0])[object_mask]
    full_query_inds = np.arange(adata.shape[0])[~object_mask]
    #for each unassigned bin, we'll find its k nearest neighbours in the assigned space
    #build a reference using the assigned bins' coordinates
    ckd = scipy.spatial.cKDTree(coords[object_mask, :])
    #query it using the unassigned bins' coordinates
    dists, hits = ckd.query(x=coords[~object_mask,:], k=k, workers=-1)
    #convert the identified indices back to the full cell space
    hits = full_reference_inds[hits]
    #get the label calls for each of the hits
    calls = labels[hits]
    #get the area (bin count) of each object
    label_values, label_counts = np.unique(labels, return_counts=True)
    if max_bin_distance is None:
        #compute the object's sphere's radius as sqrt(nbin/pi)
        #scale to radius of cell by multiplying by volume_ratio^(1/3)
        #and subtract away the original radius to account for presence of nucleus
        #do a ceiling to compensate for possible reduction of area in slice
        label_distances = np.ceil((volume_ratio**(1/3)-1) * np.sqrt(label_counts/np.pi))
        #get an array where you can index on object and get the distance
        #needs +1 as the max value of label_values is actually present in the data
        label_distance_array = np.zeros((np.max(label_values)+1,))
        label_distance_array[label_values] = label_distances
    else:
        #just use the provided value
        label_distance_array = np.ones((np.max(label_values)+1,)) * max_bin_distance
    #construct a matching dimensionality array of max distance allowed per call
    max_call_distance = label_distance_array[calls]
    #mask bins too far away from call with arbitrary high value
    dist_mask = 1000
    dists[dists > max_call_distance] = dist_mask
    #evaluate the minima in each row. start by getting said minima
    min_per_bin = np.min(dists, axis=1)[:,None]
    #now get positions in each row that have the minimum (and aren't the mask)
    is_hit = (dists == min_per_bin) & (min_per_bin < dist_mask)
    #case one - we have a solitary hit of the minimum
    clear_mask = (np.sum(is_hit, axis=1) == 1)
    #get out the indices of the bins
    clear_query_inds = full_query_inds[clear_mask]
    #np.argmin(axis=1) finds the column of the minimum per row
    #subsequently retrieve the matching hit from calls
    clear_query_labels = calls[clear_mask, np.argmin(dists[clear_mask, :], axis=1)]
    #insert calls into object
    adata.obs.loc[adata.obs_names[clear_query_inds], expanded_labels_key] = clear_query_labels
    #case two - 2+ assigned bins are equidistant
    ambiguous_mask = (np.sum(is_hit, axis=1) > 1)
    if np.sum(ambiguous_mask) > 0:
        #get their indices in the original cell space
        ambiguous_query_inds = full_query_inds[ambiguous_mask]
        if subset_pca:
            #in preparation of PCA, get a master list of all the bins to PCA
            #we've got two sets - the query bins, and their k hits
            #the hits needs to be .flatten()ed after masking to become 1d again
            #np.unique sorts in an ascending fashion, which is convenient
            smol = np.unique(np.concatenate([hits[ambiguous_mask,:].flatten(), ambiguous_query_inds]))
            #prepare a PCA as a representation of the GEX space for solving ties
            #can just run straight on an array to get a PCA matrix back. convenient!
            #keep the object's X raw for subsequent cell creation
            pca_smol = sc.pp.pca(np.log1p(adata.X[smol, :]))
            #mock up a "full-scale" PCA matrix to not have to worry about different indices
            pca = np.zeros((adata.shape[0], pca_smol.shape[1]))
            pca[smol, :] = pca_smol
        else:
            #just run a full space PCA
            pca = sc.pp.pca(np.log1p(adata.X))
        #compute the distances between the expression profiles of the undecided bin and the neighbours
        #np.linalg.norm is the fastest way to get euclidean, subtract two point sets beforehand
        #pca[hits[ambiguous_mask, :]] is bins by k by num_pcs
        #pca[ambiguous_query_inds, :] is bins by num_pcs
        #add the [:, None, :] and it's bins by 1 by num_pcs, and subtracts as you'd hope
        eucl_input = pca[hits[ambiguous_mask, :]] - pca[ambiguous_query_inds, :][:, None, :]
        #can just do this along axis=2 and get all the distances at once
        eucl_dists = np.linalg.norm(eucl_input, axis=2)
        #mask ineligible bins with arbitrary high value
        eucl_mask = 1000
        eucl_dists[~is_hit[ambiguous_mask, :]] = eucl_mask
        #define calls based on euclidean minimum
        #same argmin/mask logic as with clear before
        ambiguous_query_labels = calls[ambiguous_mask, np.argmin(eucl_dists, axis=1)]
        #insert calls into object
        adata.obs.loc[adata.obs_names[ambiguous_query_inds], expanded_labels_key] = ambiguous_query_labels

def salvage_secondary_labels(adata, primary_label="labels_he_expanded", secondary_label="labels_gex", labels_key="labels_joint"):
    '''
    Create a joint ``labels_key`` that takes the ``primary_label`` and fills in 
    unassigned bins based on calls from ``secondary_label``. Only objects that do not 
    overlap with any bins called as part of ``primary_label`` are transferred over.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Needs ``primary_key`` and ``secodary_key`` in ``.obs``.
    primary_label : ``str``, optional (default: ``"labels_he_expanded"``)
        ``.obs`` key holding the main labels. Integers, with 0 being unassigned to an 
        object.
    secondary_label : ``str``, optional (default: ``"labels_gex"``)
        ``.obs`` key holding the labels to be inserted into unassigned bins. Integers, 
        with 0 being unassigned to an object.
    labels_key : ``str``, optional (default: ``"labels_joint"``)
        ``.obs`` key to store the combined label information into. Will also add a 
        second column with ``"_source"`` appended to differentiate whether the bin was 
        tagged from the primary or secondary label.
    '''
    #these are the bins that have the primary label assigned
    primary = adata.obs.loc[adata.obs[primary_label] > 0, :]
    #these are the bins that lack the primary label, but have the secondary label
    secondary = adata.obs.loc[adata.obs[primary_label] == 0, :]
    secondary = secondary.loc[secondary[secondary_label] > 0, :]
    #kick out any secondary labels that appear in primary-labelled bins
    #we are just interested in ones that are unique to bins without primary labelling
    secondary_to_take = np.array(list(set(secondary[secondary_label]).difference(set(primary[secondary_label]))))
    #both of these labels are integers, starting from 1
    #offset the new secondary labels by however much the maximum primary label is
    offset = np.max(adata.obs[primary_label])
    #use the primary labels as a basis
    adata.obs[labels_key] = adata.obs[primary_label].copy()
    #flag any bins that are assigned to our secondary labels of interest
    mask = np.isin(adata.obs[secondary_label], secondary_to_take)
    adata.obs.loc[mask, labels_key] = adata.obs.loc[mask, secondary_label] + offset
    #store information on origin of call
    adata.obs[labels_key+"_source"] = "none"
    adata.obs.loc[adata.obs[primary_label]>0, labels_key+"_source"] = "primary"
    adata.obs.loc[mask, labels_key+"_source"] = "secondary"
    #notify of how much was salvaged
    print("Salvaged "+str(len(secondary_to_take))+" secondary labels")

def bin_to_cell(adata, labels_key="labels_expanded", spatial_keys=["spatial"], diameter_scale_factor=None):
    '''
    Collapse all bins for a given nonzero ``labels_key`` into a single cell. 
    Gene expression added up, array coordinates and ``spatial_keys`` averaged out. 
    ``"spot_diameter_fullres"`` in the scale factors multiplied by 
    ``diameter_scale_factor`` to reflect increased unit size. Returns cell level AnnData, 
    including ``.obs["bin_count"]`` reporting how many bins went into creating the cell.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts. Needs ``labels_key`` in ``.obs`` 
        and ``spatial_keys`` in ``.obsm``.
    labels_key : ``str``, optional (default: ``"labels_expanded"``)
        Which ``.obs`` key to use for grouping 2um bins into cells. Integers, with 0 being 
        unassigned to an object. If an extra ``"_source"`` column is detected as a result 
        of ``b2c.salvage_secondary_labels()`` calling, its info will be propagated per 
        label.
    spatial_keys : list of ``str``, optional (default: ``["spatial"]``)
        Which ``.obsm`` keys to average out across all bins falling into a cell to get a 
        cell's respective spatial coordinates.
    diameter_scale_factor : ``float`` or ``None``, optional (default: ``None``)
        The object's ``"spot_diameter_fullres"`` will be multiplied by this much to reflect 
        the change in unit per observation. If ``None``, will default to the square root of 
        the mean of the per-cell bin counts.
    '''
    #a label of 0 means there's nothing there, ditch those bins from this operation
    adata = adata[adata.obs[labels_key]!=0]
    #use the newly inserted labels to make pandas dummies, as sparse because the data is huge
    cell_to_bin = pd.get_dummies(adata.obs[labels_key], sparse=True)
    #take a quick detour to save the cell labels as they appear in the dummies
    #they're likely to be integers, make them strings to avoid complications in the downstream AnnData
    cell_names = [str(i) for i in cell_to_bin.columns]
    #then pull out the actual internal sparse matrix (.sparse) as a scipy COO one, turn to CSR
    #this has bins as rows, transpose so cells are as rows (and CSR becomes CSC for .dot())
    cell_to_bin = cell_to_bin.sparse.to_coo().tocsr().T
    #can now generate the cell expression matrix by adding up the bins (via matrix multiplication)
    #cell-bin * bin-gene = cell-gene
    #(turn it to CSR at the end as somehow it comes out CSC)
    X = cell_to_bin.dot(adata.X).tocsr()
    #create object, stash stuff
    cell_adata = ad.AnnData(X, var = adata.var)
    cell_adata.obs_names = cell_names
    #need to bust out deepcopy here as otherwise altering the spot diameter gets back-propagated
    cell_adata.uns['spatial'] = deepcopy(adata.uns['spatial'])
    #getting the centroids (means of bin coords) involves computing a mean of each cell_to_bin row
    #premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    #use that to divide each row by it sum (.sum(axis=1)), then matrix multiply the result by bin coords
    #stash the sum into a separate variable for subsequent object storage
    #cell-cell * cell-bin * bin-coord = cell-coord
    bin_count = np.asarray(cell_to_bin.sum(axis=1)).flatten()
    row_means = scipy.sparse.diags(1/bin_count)
    cell_adata.obs['bin_count'] = bin_count
    #take the thing out for a spin with array coordinates
    cell_adata.obs["array_row"] = row_means.dot(cell_to_bin).dot(adata.obs["array_row"].values)
    cell_adata.obs["array_col"] = row_means.dot(cell_to_bin).dot(adata.obs["array_col"].values)
    #generate the various spatial coordinate systems
    #just in case a single is passed as a string
    if type(spatial_keys) is not list:
        spatial_keys = [spatial_keys]
    for spatial_key in spatial_keys:
        cell_adata.obsm[spatial_key] = row_means.dot(cell_to_bin).dot(adata.obsm[spatial_key])
    #of note, the default scale factor bin diameter at 2um resolution stops rendering sensibly in plots
    #by default estimate it as the sqrt of the bin count mean
    if diameter_scale_factor is None:
        diameter_scale_factor = np.sqrt(np.mean(bin_count))
    #bump it up to something a bit more sensible
    library = list(adata.uns['spatial'].keys())[0]
    cell_adata.uns['spatial'][library]['scalefactors']['spot_diameter_fullres'] *= diameter_scale_factor
    #if we can find a source column, transfer that
    if labels_key+"_source" in adata.obs.columns:
        #hell of a one liner. the premise is to turn two columns of obs into a translation dictionary
        #so pull them out, keep unique rows, turn everything to string (as labels are strings in cells)
        #then set the index to be the label names, turn the thing to dict
        #pd.DataFrame -> dict makes one entry per column (even if we just have the one column here)
        #so pull out our column's entry and we have what we're after
        mapping = adata.obs[[labels_key,labels_key+"_source"]].drop_duplicates().astype(str).set_index(labels_key).to_dict()[labels_key+"_source"]
        #translate the labels from the cell object
        cell_adata.obs[labels_key+"_source"] = [mapping[i] for i in cell_adata.obs_names]
    return cell_adata