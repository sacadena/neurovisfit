# neurovisfit

[![Release](https://img.shields.io/github/v/release/sacadena/neurovisfit)](https://img.shields.io/github/v/release/sacadena/neurovisfit)
[![Build status](https://img.shields.io/github/actions/workflow/status/sacadena/neurovisfit/main.yml?branch=main)](https://github.com/sacadena/neurovisfit/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sacadena/neurovisfit/branch/main/graph/badge.svg)](https://codecov.io/gh/sacadena/neurovisfit)
[![Commit activity](https://img.shields.io/github/commit-activity/m/sacadena/neurovisfit)](https://img.shields.io/github/commit-activity/m/sacadena/neurovisfit)
[![License](https://img.shields.io/github/license/sacadena/neurovisfit)](https://img.shields.io/github/license/sacadena/neurovisfit)

Code to reproduce the results of Cadena et al. 2024 PlosCB to predict macaque V1 and V4 responses to natural
images.

Download 
[V1](https://figshare.com/articles/dataset/Monkey_V1_responses_to_natural_images_from_Cadena_et_al_2023/23056805?backTo=/collections/Monkey_V1_and_V4_single-cell_responses_to_natural_images_ephys_Data_from_Cadena_et_al_2024_/6658331) and [V4](https://figshare.com/articles/dataset/Monkey_V4_responses_to_natural_images_from_Cadena_et_al_2023/23060540?backTo=/collections/Monkey_V1_and_V4_single-cell_responses_to_natural_images_ephys_Data_from_Cadena_et_al_2024_/6658331) datasets to a local folder


```python

# Get data loaders
from pathlib import Path
from neurovisfit.data.loaders import get_dataloaders
dataset_name = "v4_data_cadena_et_al_2024"
data_path = Path("your_data_path/v4_data")
dataloaders = get_dataloaders(dataset_name, data_path)


# Build model
from neurovisfit.models.builder import build_model
model_name = "v4__core_resnet50_l2_01_layer_3_0__readout_gauss"
model = build_model(
    model_name,
    dataloaders=dataloaders,
    seed=42,
)

# Train model
from neurovisfit.trainers.params import get_trainer_params_from_config
from neurovisfit.trainers.trainer import train_and_evaluate

trainer_params = get_trainer_params_from_config('standard_trainer')
results = train_and_evaluate(
    model=model,
    dataloaders=dataloaders,
    params=trainer_params,
    seed=42,
    device='cuda',
)
```

## Citation
```commandline
@article{cadena2022diverse,
 title={Diverse task-driven modeling of macaque V4 reveals functional specialization towards semantic tasks},
 author={Cadena, Santiago A and Willeke, Konstantin F and Restivo, Kelli and Denfield, George and Sinz, Fabian H and Bethge, Matthias and Tolias, Andreas S and Ecker, Alexander S},
 journal={bioRxiv},
 pages={2022--05},
 year={2022},
 publisher={Cold Spring Harbor Laboratory}
}
```