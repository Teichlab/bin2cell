.. Bin2cell documentation master file, created by
   sphinx-quickstart on Thu May 16 12:40:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bin2cell
========

Project home page `here <https://github.com/Teichlab/bin2cell>`_, demonstration notebook `here <https://nbviewer.org/github/Teichlab/bin2cell/blob/main/notebooks/demo.ipynb>`_.

Main workflow functions
-----------------------
.. autosummary::
   :toctree:
   
   bin2cell.scaled_he_image
   bin2cell.destripe
   bin2cell.grid_image
   bin2cell.stardist
   bin2cell.insert_labels
   bin2cell.expand_labels
   bin2cell.salvage_secondary_labels
   bin2cell.bin_to_cell

Utility functions
-----------------
.. autosummary::
   :toctree:
   
   bin2cell.get_crop
   bin2cell.view_stardist_labels
   bin2cell.load_image
   bin2cell.check_array_coordinates
   bin2cell.mpp_to_scalef
   bin2cell.get_mpp_coords
   bin2cell.destripe_counts
