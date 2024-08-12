# Changelog

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
