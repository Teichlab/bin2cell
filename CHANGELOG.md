# Changelog

## 0.3.4
- StarDist moved to optional dependency group, global import removed
- `b2c.stardist()` accepts optional custom model/axes, and prints out parameter overrides

## 0.3.3
- `b2c.view_cell_labels()` to show cell-level metadata on the morphology segmentation

## 0.3.2
- `b2c.view_labels()` as a more lightweight, whole image level take on `b2c.view_stardist_labels()`
- `b2c.actual_vs_inferred_image_shape()` as an image dimension based assessment of source image validity
- custom image functions now can skip storing the image in the object, open up control over the image key, have the buffer included by default in both sets of generated keys, and print out any stored keys
- `b2c.expand_labels()` switches algorithm control to new `algorithm` argument
- `b2c.salvage_secondary_labels()` stores secondary label offset
- `b2c.bin_to_cell()` stores object ID as an integer in the cell object `.obs`

## 0.3.1
- add `b2c.check_bin_image_overlap()` for friendlier handling of users loading the incorrect image

## 0.3.0
- rework `b2c.expand_labels()` to be more robust:
    - evaluate a user-controlled `k` assigned bins for each unassigned bin
    - alternative expansion distance algorithm based on label area
    - `max_call_distance` is a new query by `k` array with maximum acceptable distance for each hit, making potential algorithmic expansion easier
    - simplify minima evaluation logic into two (queries with 1 and >1) rather than three steps

## 0.2.0
- add `spaceranger_image_path` to `b2c.read_visium()` in response to 10X creating a unified spaceranger spatial folder
- `b2c.scaled_if_image()` for processing IF images
- ignore out of bounds pixels in `b2c.insert_labels()`

## 0.1.1
- cap `numpy` at 1.x for the time being

## 0.1.0
- initial release
