# Example based sampling with diffusion models

Repository for the paper : Example based sampling with diffusion models. 

```
Bastien Doignies, Nicolas Bonneel, David Coeurjolly, Julie Digne, Lois Paulin,
Jean-Claude Iehl, and Victor Ostromoukhov. 2023. Example-Based Sampling
with Diffusion Models. In SIGGRAPH Asia 2023 Conference Papers (SA Con-
ference Papers ’23), December 12–15, 2023, Sydney, NSW, Australia. ACM, New
York, NY, USA, 11 pages. https://doi.org/10.1145/3610548.361824
```

## Preparing dataset

Preparing the dataset can be done with the `QMCDatabaseBuilder` utility class. Its goal is to parse a folder and to create the database with all preprocessed data. It handles multiple shapes inputs automatically.

```python
from data.DatasetBuilder import QMCDatabaseBuilder
from data.Transforms import to_image_optimal_transport

def properties_func(group_name, points):
    """
        Conditionning values, will be call once
        for each example in the database. 

        group_name: str
            Name of the group 
        points: np.array
            Single example of the training set, shape
            depends on the loader function

        Returns:
            list of values that serves as conditionning for the network
    """
    cond_values = {
        "SOT":  [1, 0, 0], 
        "LDBN": [0, 1, 0], 
        "GBN":  [0, 0, 1]
    }
    # May also depends on points, sh
    return cond_values.get(group_name, [0, 0, 0 ])

def group_by_name(name):
    """
        Groupe file by name. 

        name: str
            Path to an example file
        
        Returns:
            String that identifies the group of the file
    """
    # Suppose path are : /path/to/file/{class}_{i}.dat
    # and that group is {class}
    return name.split('_')[0]


list_of_directories = [] # To fill

# Only scan files to process for now
builder = QMCDatabaseBuilder(
    # Directories to scan
    list_of_directories, 
    # Output filename
    "output.hdf5",
    # For conditionning
    properties=properties_func,
    # Transform inputs, here default mapping to grid with optimal transport 
    transform=to_image_optimal_transport, 
    # Group by name
    group=group_by_name
    # Data loader
    loader=np.loadtxt
)

# Perform loading / property computation / data transformation for each file found
builder.build()
```

It outputs a hdf5 file :

```
output.hdf5
├── Group_1/
│   ├── Scale_1/
│   │   ├── data
│   │   ├── data_t
│   │   └── prop
│   ├── Scale_2/
│   │   ├── data
│   │   ├── data_t
│   │   └── prop
│   └── ...
├── Group_2/
│   ├── Scale_1/
│   │   └── ...
│   ├── Scale_2/
│   │   └── ...
│   └── ...
└── ...
```

Where:

* `data`   are the original training examples of shape (nb_examples, N, D)
* `data_t` are the transformed examples of shape (nb_examples, transformed_shape)
* `prop`   are the conditionning values of shape (nb_examples, conditionning_shape)


## Training a model

A model is defined by its configuration file. It is a json file whose template may follow : 

```json
{
    "path": "PATH_TO_OUTPUT",
    "model": 
    {
        "num_channels": 2,
        "ch": 128, 
        "out_ch": 2, 
        "ch_mult": [1, 2, 3],
        "num_res": 2, 
        "attn_layers": [],
        "attn_middle": false,
        "dropout": 0.1, 
        "resamp_with_conv": true, 
        "cond": {
            "model": "id", 
            "select": ["0:9"],
            "model_out": 9,
            "embd_dim": 16
        }
    },
    "diffusion": 
    {
        "betas": {
            "min": 1e-4,
            "max": 1e-2, 
            "count": 1000,
            "schedule": "linear"
        },
        "loss": "mse" 
    },
    "train": 
    {
        "dataset": "output.hdf5",
        "samplers": ["GROUP_1", "GROUP_2", "..."], 
        "batch_size": 32,
        "ema_decay": 0.995,
        "ema_update_every": 25,
        "lr": 1e-5
    },
    "eval": 
    {    
        "directory": "eval/", 
        "freq": 1000000,
        "ts": [50, 1000]
    }
}
```

Most parameters are self explanatory. 

For the others : 

* `num_channels`: is the input data dimension
* `out_ch`: is the output data dimension
* `ch`: is the base channel count
* `ch_mult`: is the number of channel multiplier. It also defines the UNET depth
* `attn_layers`: is a list of depth where attention should be applied
* `attn_middle': boolean to control if attention should bottom-most layer
* `cond`: 
    * `model`: only id supported
    * `select`: allows to perform subselection of properties. May use slicing as string
    * `model_out`: number of conditionning values
    * `embd_dim`: embedding dimension for conditionning values
* `eval`:
    * `directory`: sub directory to put evaluation folder
    * `freq`: frequency of evaluation (in optimization steps)
    * `ts`: number of diffusion timesteps to use in eval

Once a model is defined, run : 

* python train.py -c config.json

Optionnally, you can provide some arguments to control training times: 

```bash
usage: python train.py [-h] -c CONFIG [--its ITS] [-t TIME] [--tqdm TQDM]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to config file (default: None)
  --its ITS             Number of optimization steps (default: 1e10)
  -t TIME, --time TIME  Training time in minutes (default: 180)
  --tqdm TQDM           Adds progress bar for training (default: True)
```

## Sampling from the model

To sample from the model, follow the script `sample.py`.

```bash
usage: python sample.py [-h] -c CONFIG -m MODEL [--cond COND [COND ...]] [-s SHAPE [SHAPE ...]] [-o OUTPUT] [-t TIMESTEPS] [--ot OT]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Config file
  -m MODEL, --model MODEL
                        Path to checkpoint file
  --cond COND [COND ...]
                        Conditionning values
  -s SHAPE [SHAPE ...], --shape SHAPE [SHAPE ...]
                        Shape of points to generate
  -o OUTPUT, --output OUTPUT
                        Output file
  -t TIMESTEPS, --timesteps TIMESTEPS
                        Number of timesteps
  --ot OT               Applied inverse ot mapping transform to input
```

Example : 

```
python sample.py -c path_to_config.json -m path_to_model.ckpt -s batch dimension n1 ... nd --cond c1 c2 ... ck
```

Where n1 x n2 x ... x nd is the number of point to sample and c1 c2 ... ck are the condtionnion values for the batch.

## Requirements 

For sampling from the model, the required libraries are :

* pytorch
* matplotlib
* tqdm

A conda environment can be created and activated with the `environment_sample.yml` file : 

```bash
conda env create -f environment_sample.yml
conda activate qmcdiffusion_sample
```

For training, a few more libraries are required : 

* tensorboardX
* ema-pytorch
* POT
* scipy
* h5py

The complete train environment can be created and activated with the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate qmcdiffusion
```

## Checkpoints and databases

Trained models and database can be found [here](https://projet.liris.cnrs.fr/qmcdiffusion/). Sampling and example usage
for trained models can be found in the `sample_example.sh` script. When run (`bash sample_example.sh`), it downloads a 
model and sample a single pointset from it.