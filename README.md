# GT-NeRF
This is the final project for the course ***Computer Vision*** taught by Prof. Pengshuai Wang at Peking University. In this project, we intend to implement the classic model of [NeRF](http://www.matthewtancik.com/nerf) with positional encoding and fit on multi-view images. Besides, we will also implement the [extension of NeRF which is capable of modeling dynamic scenes](https://www.albertpumarola.com/research/D-NeRF/index.html) utilizing both the straightfoward way of representing the scene by a 6D input of the query 3D location, viewing direction and time $(x, y, z, t, θ, ϕ)$ and the method with two neural network modules $Ψ_t,  Ψ_x$ (proposed by [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html)).  

## File Structure
```
├── logs                    # save checkpoints and pre-trained weights
├── data                    # data for training and testing
├── configs                 # parameters for training and testing, for adjusting different models
├── requirements.txt        # environment required to run the code
├── run.py                  # code
├── ...
```

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
|   |   ├── Lego
|   |   ├── Fern
|   |   ├── ...
|   ├── D-NeRF
│   |   ├── standup 
│   |   ├── mutant
|   |   ├── lego
```

## Download Dataset[1][2]
 You can download the datasets from [drive](https://drive.google.com/drive/folders/1Zy0wkFIy7EApZiJEEVjQVLYYLk9_R2EL?usp=sharing). Unzip the downloaded data to the project root dir in order to train. See the following directory structure for an example:
```
├── data 
│   ├── NeRF
|   |   ├── Lego
|   |   ├── Fern
|   |   ├── ...
|   ├── D-NeRF
│   |   ├── standup 
│   |   ├── mutant
|   |   ├── lego
|   |   ├── ...
```
## Configs
These .txt files are the basic parameters for training, loading data and rendering. You can adjust by yourself to change model.
```
├── configs
│   ├── NeRF
|   |   ├── Lego.txt
|   |   ├── Fern.txt
|   |   ├── ...
|   ├── D-NeRF
│   |   ├── standup.txt 
│   |   ├── mutant.txt
|   |   ├── lego.txt
|   |   ├── ...
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
python run.py --config configs/D-NeRF/mutant.txt --render_only --render_test
```
This command will run the `mutant` experiment. When finished, results are saved to `./D-NeRF/logs/mutant/renderonly_test_799999` To quantitatively evaluate model run `metrics.ipynb` notebook

## Train
First download the dataset. Then,  
1. For NeRF:
```
python run.py --config configs/NeRF/Lego.txt
```
2. For D-NeRF:
```
python run.py --config configs/D-NeRF/mutant.txt
```
3. For T-NeRF:
```
python run.py --config configs/D-NeRF/mutant.txt --is_straightforward True
```
4. For GT-NeRF:
```
python run.py --config configs/D-NeRF/mutant.txt --is_ViT True
```

## Segregation of Duties
This project is all done by me[@Kuangzhi Ge](https://github.com/MAMBA4L924) and [@Yiyang Tian](https://github.com/Jappwhagg).
Yiyang is mainly resiponsible for: 
  1) the implementation of positional encoding 
  2) the implementation of NeRF model
  3) LLFF NeRF:PM_Model
I am responsible for:
  1) the implementation of D-NeRF, T-NeRF
  2) propose GT-NeRF
And we are the co-authors of the final report for the final project.

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
