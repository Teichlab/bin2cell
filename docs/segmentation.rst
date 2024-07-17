StarDist segmentation
=====================

The utility and performance of specific StarDist models (H&E or fluorescence) is highly dependent on the input image resolution. The models are tuned for a specific size-range nuclei and as such if the number of pixels per nuclei would be too high ("high resolution image") the model might find small structures as nucleoli. In a similar sense, if the number of pixels per nuclei is too low ("low resolution image") the model might find aggregates of cells or other structures as nuclei. For this reason we would recommend using the ``mpp`` (microns per pixel) range of 0.2-0.5 and test it on your data, adjusting as necessary to ensure the detected objects are nuclei. Lower ``mpp`` (higher image resolution) would sometimes lead to more accurate segmentations with the cost of speed, while larger mpp will perform fast but might be less accurate.

The other important parameters of the StarDist model are (1) the object probability threshold ``prob_thresh`` which is the cutoff of inclusion of a segmented object in the nuclei prediction. A lower cutoff would include more cells but might lead to more false positives. (2) ``nms_thresh`` tells the model what is the expected overlap between objects. If the nuclei are very dense they would also be expected to overlap in the image.

Example 1 - ``mpp``
-------------------

Example 1

Example 2 - ``prob_thresh``
---------------------------

Example 2