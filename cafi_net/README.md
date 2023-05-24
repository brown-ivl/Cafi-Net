# Training and Testing Cafinet 

This README contains instructions to train and test the Cafinent.



## Loading the environment

Make sure you have Anaconda or Miniconda installed before you proceed to load this environment.

```bash
# Creating the conda environment and loading it
conda env create -f environment.yml
conda activate Cafinet_torch
```

Training and testing .

#### Training

1. In `configs/Canonical_fields.yaml` change the dataset path to the downloaded dataset.

```
# In configs/Canonical_fields.yaml
dataset:
  dataset_path: <change path to to training dataset>
val_dataset:
  dataset_path:: <change path to to validation dataset>
```

2. Run the code below

```bash
# Run the code to train
CUDA_VISIBLE_DEVICES=0 python main.py
```

#### Testing

1. The test script tests the model on the validation set and saves the output as ply files for visualization.

```bash
# Test the trained model
# weight files are stored at path outputs/<date_of_run_train>/<time_of_run_train>/checkpoints/ 
CUDA_VISIBLE_DEVICES=0 python3 tester.py 'test.weights="<model_weights_path>"' 'test.skip=1'
```

2. After running the test script you will find a new directory with stored pointclouds at location `outputs/<date_of_run_test>/<time_of_run_test>/pointclouds/`
