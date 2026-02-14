# U-Net: Convolutional Networks for Biomedical Image Segmentation Replication
This is my replication of U-Net Model on the ISBI 2012 Challenge Dataset

## Setup and Running
This project was created using `uv` and is highly recommended<br>
After installing `uv` this project should run out of the box<br>

### Data Setup
You can get the Segmentation of neuronal structures in EM stacks challenge - ISBI 2012 Dataset from the [Challenge Webpage](https://imagej.net/events/isbi-2012-segmentation-challenge)

Once this is downloaded you will need to run the `process_images()` script to create our datasets<br>
```
uv run process_images.py
```

### Training
Before kicking off training you should update the weights and biases variables in train.py to match your account.<br>
If not using Weights and Biases (not recommended) you can set logs to `False` in main.py<br>
To kick off training you can run<br>
```uv run main.py```

### Deviations from the Original Paper
Resolving Input-Output Size Mismatches
- The original paper has an input size of 572x572 and and output size of 388x388 and uses a tiling approach to resolve the discrepancies at inference time
- This implementation uses padding to keep the input and outputs the same size. This is a more modern approach and produces a more elegant solution 

# Citation 
[Paper](https://arxiv.org/abs/1505.04597)

```bibtex
@article{DBLP:journals/corr/RonnebergerFB15,
  author       = {Olaf Ronneberger and
                  Philipp Fischer and
                  Thomas Brox},
  title        = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  journal      = {CoRR},
  volume       = {abs/1505.04597},
  year         = {2015},
  url          = {http://arxiv.org/abs/1505.04597},
  eprinttype    = {arXiv},
  eprint       = {1505.04597},
  timestamp    = {Mon, 13 Aug 2018 16:46:52 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/RonnebergerFB15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```