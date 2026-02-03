# U-Net: Convolutional Networks for Biomedical Image Segmentation Replication

## Data
To replicate we chose to replicate the results on the Segmentation of neuronal structures in EM stacks challenge - ISBI 2012 [Challenge Webpage](https://imagej.net/events/isbi-2012-segmentation-challenge)

## Deviations from the Original Paper
Resolving Input-Output Size Mismatches
- The original paper has an input size of 572x572 and and output size of 388x388 and uses a tiling approach to resolve the discrepancies at inference time
- This implementation uses padding to keep the input and outputs the same size. This is a more modern approach and produces a more elegant solution 