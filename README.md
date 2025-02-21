# Sparse Activations as Conformal Predictors".

Repository containing code to reproduce the results of the paper "[Sparse Activations as Conformal Predictors](https://arxiv.org/abs/2502.14773)".

## Setup 

Create environment with Python version `3.11.6` and install `requirements.txt`.

Install package **confpred** by running the following on the home directory of the project: 

`pip install .`

## Basic Usage

### Training Models and Finetuning

Example code to train/fine-tune models can be found in `example_usage/train.py`. 

Types of models supported:
 - `'cnn'` - convolution neural network trained from scratch (`confpred/classifier/CNN.py`)
 - `'vit` - finetuning of [Google's Vision Transformer model](https://huggingface.co/google/vit-base-patch16-224) (`confpred/classifier/FineTuneVit.py`)
 - `'bert'` - finetuning of [Bert model](https://huggingface.co/google-bert/bert-base-uncased) (`confpred/classifier/FineTuneBertForSequenceClassification.py`)

Supported datasets:`'CIFAR10'`, `'CIFAR100'`,`'ImageNet'` and `'NewsGroups'`. 
Supported losses for training: `'softmax'` (standard log likelihood loss), `'entmax'` (1.5 entmax loss) and `'sparsemax'` (sparsemax loss).

One can download the logits of the already trained models used for the analysis presented in the report in the [drive](https://drive.google.com/drive/folders/1L5RIPNEUwzYsH__EfVQct9-zeuTX0CuE?usp=drive_link).

### Conformal Predictors

To find the optimal RAPS parameters, run `scripts/optimal_raps_parameters.py` and to find optimal *opt-entmax* parameter run `scripts/optimal_entmax_params.py`. Alternatively, download optimal parameter files from [drive](https://drive.google.com/drive/folders/1aejfKdLl--kQo6nvLpqEJNWbs01S4d2G?usp=drive_link).

The notebook `notebooks/all_methods_cp.ipynb` applies conformal prediction over five different splits of the data, as described in the paper, and writes: set prediction arrays in folder `data/set_prediction` and a table with all coverage and average set size results (also available [here](https://drive.google.com/drive/folders/1aejfKdLl--kQo6nvLpqEJNWbs01S4d2G?usp=drive_link)).

## Reproducing paper results

 - Coverage and average set size analysis can be found in `notebooks/coverage_set_analysis.ipynb` 
 - Adaptiveness and coverage by set size analysis can be found in `notebooks/set_size_coverage.ipynb`