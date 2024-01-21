# GT-NeRF
This is the final project for the course ***Computer Vision*** taught by Prof. Pengshuai Wang at Peking University. In this project, we intend to implement the classic model of [NeRF](http://www.matthewtancik.com/nerf) with positional encoding and fit on multi-view images. Besides, we will also implement the [extension of NeRF which is capable of modeling dynamic scenes](https://www.albertpumarola.com/research/D-NeRF/index.html) utilizing both the straightfoward way of representing the scene by a 6D input of the query 3D location, viewing direction and time $(x, y, z, t, θ, ϕ)$ and the method with two neural network modules $Ψ_t,  Ψ_x$ (proposed by [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html)).  

## Installation
```
git clone https://github.com/MAMBA4L924/GT-NeRF.git
cd GT-NeRF
pip install -r requirements.txt
cd ..
```

## Download Pre-trained Weights[1]
 You can download the pre-trained models from [drive](https://drive.google.com/drive/folders/1p3HAV7irfNl3UQyI0R1nr68mxngmZuDk?usp=sharing). Unzip the downloaded data to the project root dir in order to test it later. See the following directory structure for an example:
```
├── logs 
│   ├── NeRF
|   |   ├── lego_NeRF
|   |   ├── ficus
|   ├── D-NeRF
│   |   ├── standup 
│   |   ├── mutant
|   |   ├── lego_N
```

## Download Dataset[1][2]
 You can download the datasets from [drive](https://drive.google.com/drive/folders/1Zy0wkFIy7EApZiJEEVjQVLYYLk9_R2EL?usp=sharing). Unzip the downloaded data to the project root dir in order to train. See the following directory structure for an example:
```
├── logs 
│   ├── NeRF
|   |   ├── lego_N
|   |   ├── ficus
|   ├── D-NeRF
│   |   ├── standup 
│   |   ├── mutant
|   |   ├── lego_N
```

## Demo[1]
We provide simple jupyter notebooks to explore the model. To use them first download the pre-trained weights and dataset.

| Description      | Jupyter Notebook |
| ----------- | ----------- |
| Synthesize novel views at an arbitrary point in time. | render.ipynb|
| Reconstruct mesh at an arbitrary point in time. | reconstruct.ipynb|
| Quantitatively evaluate trained model. | metrics.ipynb|

## Test
First download pre-trained weights and dataset. Then, 
```
python run.py --config configs/mutant.txt --render_only --render_test
```
This command will run the `mutant` experiment. When finished, results are saved to `./logs/mutant/renderonly_test_799999` To quantitatively evaluate model run `metrics.ipynb` notebook

## Train
First download the dataset. Then,  
1. For NeRF:
```
python run.py --config configs/lego_N.txt
```
or
```
python run.py --config configs/ficus.txt
```
2. For D-NeRF:
```
python run.py --config configs/mutant.txt
```
## Citations
[1] @article{pumarola2020d,
  title={D-NeRF: Neural Radiance Fields for Dynamic Scenes},
  author={Pumarola, Albert and Corona, Enric and Pons-Moll, Gerard and Moreno-Noguer, Francesc},
  journal={arXiv preprint arXiv:2011.13961},
  year={2020}
}  

[2] @misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
