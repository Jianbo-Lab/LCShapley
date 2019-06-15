# 

Code for L-Shapley and C-Shapley in the paper [L-Shapley and C-Shapley: Efficient Model Interpretation for Structured Data](https://arxiv.org/pdf/1808.02610.pdf) by Jianbo Chen, Le Song, Martin J. Wainwright, Michael I. Jordan. 

## Dependencies
The code runs with Python 2.7 and requires Tensorflow of version 1.1 or higher. Please `pip install` the following packages:
- `numpy`
- `pandas`
- `keras`
- `tensorflow` 
- `csv`

## Running in Docker, MacOS or Ubuntu
We provide as an example the source code to run CCM on the three synthetic datasets in the paper. Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/Jianbo-Lab/LCShapley
cd LCShapley/texts/
############################################### 
# L-Shapley
python explain.py --method localshapley

# C-Shapley
python explain.py --method connectedshapley
```

The importance scores for each word will be saved for the first 100 test samples in IMDB movie review.

## Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/pdf/1808.02610.pdf):
```
@inproceedings{
chen2018lshapley,
title={L-Shapley and C-Shapley: Efficient Model Interpretation for Structured Data},
author={Jianbo Chen and Le Song and Martin J. Wainwright and Michael I. Jordan},
booktitle={International Conference on Learning Representations},
year={2019},
}
```