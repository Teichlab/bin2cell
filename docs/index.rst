.. Bin2cell documentation master file, created by
   sphinx-quickstart on Thu May 16 12:40:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bin2cell
========

Project home page `here <https://github.com/Teichlab/bin2cell>`_, demonstration notebook `here <https://bin2cell.readthedocs.io/en/latest/notebooks/demo.html>`_.

Main workflow functions
-----------------------
.. autosummary::
   :toctree:
   
   bin2cell.scaled_he_image
   bin2cell.scaled_if_image
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
   bin2cell.view_labels
   bin2cell.view_cell_labels
   bin2cell.load_image
   bin2cell.check_array_coordinates
   bin2cell.actual_vs_inferred_image_shape
   bin2cell.mpp_to_scalef
   bin2cell.get_mpp_coords
   bin2cell.overlay_onto_img
   bin2cell.destripe_counts

Obsoleted functions
-------------------
.. autosummary::
   :toctree:
   
   bin2cell.view_stardist_labels
   bin2cell.check_bin_image_overlap