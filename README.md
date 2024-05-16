# Bin2cell

Visium HD captures gene expression data at a subcellular 2um resolution. It should be possible to use this data to reconstruct cells more accurately than just using the next resolution up (8um), especially if additionally using the available high resolution morphology images.

Bin2cell proposes 2um bin to cell groupings based on segmentation, which can be done on the H&E image and/or a visualisation of the gene expression. The package also corrects for a novel technical effect in the data stemming from variable bin dimensions. The end result is an object with cells, created from grouped 2um bins assigned to the same object after segmentation, carrying spatial information and sharper H&E images for visualisation. More details in the [demo notebook](https://nbviewer.org/github/Teichlab/bin2cell/blob/main/notebooks/demo.ipynb).

## Installation

```bash
pip install bin2cell
```

Additionally, TensorFlow needs to be installed for [StarDist](https://github.com/stardist/stardist) to perform segmentation. The CPU version (installed via `pip install tensorflow`) should suffice for the scale of work performed here.

## Usage and Documentation

Please refer to the [demo notebook](https://nbviewer.org/github/Teichlab/bin2cell/blob/main/notebooks/demo.ipynb). Function docstrings are available on [ReadTheDocs](https://bin2cell.readthedocs.io/en/latest/).
