# Bin2cell

Visium HD captures gene expression data at a subcellular 2um resolution. In principle, it should be possible to use this data to reconstruct cells more accurately than just using the next resolution up (8um).

Bin2cell proposes 2um bin to cell groupings based on segmentation, which can be done on the H&E image or a visualisation of the gene expression. The package also corrects for a novel technical effect in the data stemming from variable bin dimensions.

## Installation

```bash
pip install git+https://github.com/Teichlab/bin2cell.git
```

Additionally, TensorFlow needs to be installed for [StarDist](https://github.com/stardist/stardist) to perform segmentation. The CPU version (installed via `pip install tensorflow`) should suffice for the scale of work performed here.

## Usage and Documentation

Please refer to the [demo notebook](notebooks/demo.ipynb). Docstrings detailing the arguments of the various functions can be found in the source code, and will be rendered on ReadTheDocs once the package is public.
