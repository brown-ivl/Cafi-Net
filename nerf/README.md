# NeRF (PyTorch Version)
We outline the steps to train NeRF models, to generate density fields, to visualize the generated fields and to visualize the canonical fields.

## Installing Libraries
There are two ways to set up the required libraries for running this code. However, we strongly recommend the second way of using the anaconda environment file.

### 1. Using pip
```
pip install -r requirements.txt
```

### 2. Using miniconda/anaconda (recommended)
Make sure you have Anaconda or Miniconda installed before you proceed to setup this environment.
```
# Creating the conda environment and loading it
conda env create -f environment.yml
conda activate nerf
```

## Dataset
Download the 1300 trained NeRF weights/models dataset from the below table:

| Dataset                                   | Link                                                         | Size (GB) |
| ----------------------------------------- | ------------------------------------------------------------ | --------- |
| Part 1 (4 categories)  | [link](https://nerf-fields.s3.amazonaws.com/final_nerf_models_cleaned/part1.zip) | 9.0       |
| Part 2 (4 categories)                            | [link](https://nerf-fields.s3.amazonaws.com/final_nerf_models_cleaned/part2.zip) | 9.6        |
| Part 3 (5 categories)                          | [link](https://nerf-fields.s3.amazonaws.com/final_nerf_models_cleaned/part3.zip) | 10.7        |

Note that we have divided the dataset into three parts where each part contains all the models for different categories. For most models we have included the weights after 200 epochs and 400 epochs of training. We are still working on releasing a few more of the missing models/instances for some of the categories so this is not the complete 1300 trained NeRF weights dataset but this release contains most of the models (1000+).

## Training NeRF
1. Change the "datadir" appropriately in "configs/brics.txt".
2. Run the following code:
```
python run_nerf.py --config configs/brics.txt
```

## Generating Density Fields from NeRF Models
1. In "configs/brics_sigmas.txt", if "multi_scene" is True, appropriately set the "root_dir", otherwise set the "ft_path", "basedir" and "expname" appropriately. In both cases, set the "datadir" to the appropriate path containing the poses information.
2. Run the following code:
```
python run_nerf.py --config configs/brics_sigmas.txt
```

## Visualizing Density Fields
Set the input directory appropriately in the command below:
```
python density_vis.py --input <insert path here>
```

Note that you can give a base directory containing multiple trained NeRF models as well, the above command will visualize all the models' density fields.

## Visualizing Gradient Fields
Set the input directory appropriately in the command below:
```
python grad_vis.py --input_dir <insert path here>
```

Note that you can give a base directory containing multiple trained NeRF models as well, the above command will visualize all the models' gradient fields.

## Rendering NeRF Models from Canonical Frame
1. In "configs/brics_canonical.txt", if "multi_scene" is True, appropriately set the "root_dir", otherwise set the "ft_path", "basedir" and "expname" appropriately. In both cases, set the "datadir" to the appropriate path containing the poses information, set the "category" and set the "canonical_path" to the appropriate path containing the canonical rotation information.
2. Run the following code:
```
python run_nerf.py --config configs/brics_canonical.txt
```

The generated canonical renderings will be saved in the "canonical_renderings" folder automatically.
