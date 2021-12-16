[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg)](https://shields.io/)
![PyPI version](https://badge.fury.io/py/fourier-neural-operator.svg)

# Fourier Neural Operator package

The original code and package come from : https://github.com/zongyi-li/fourier_neural_operator (the original author of the fourier neural operator paper).

We created some minor modification on the package to create a proper pip package using fourier neural operator.

You can install it using (after having download the repo)

```bash
python setup.py install
```

or simply using pypi :

```bash
pip install fourier-neural-operator
```

Then to create a fourier model with the pytorch framework, you can write :


```python
import fourier_neural_operator.fourier_2d as fourier_2d 
model = fourier_2d.FNO2d(modes1=modes1, modes2=modes2,  width=width, channel_input=3, output_channel=3)
```

You can also simply import fourier layer : 


```python
import fourier_neural_operator.fourier_2d as fourier_2d 
spectral_layer = fourier_2d.SpectralConv2d(width, width, modes1, modes2)
```

The package is still under construction and modification will come for fourier_3d and 1d.

# Fourier Neural Operator explaination

This repository contains the code for the paper:
- [(FNO) Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

In this work, we formulate a new neural operator by parameterizing the integral kernel directly in Fourier space, allowing for an expressive and efficient architecture. 
We perform experiments on Burgers' equation, Darcy flow, and the Navier-Stokes equation (including the turbulent regime). 
Our Fourier neural operator shows state-of-the-art performance compared to existing neural network methodologies and it is up to three orders of magnitude faster compared to traditional PDE solvers.

It follows from the previous works:
- [(GKN) Neural Operator: Graph Kernel Network for Partial Differential Equations](https://arxiv.org/abs/2003.03485)
- [(MGKN) Multipole Graph Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2006.09535)

You can check the code in the exemples_paper/ directory.


## Requirements
- We have updated the files to support [PyTorch 1.8.0](https://pytorch.org/). 
Pytorch 1.8.0 starts to support complex numbers and it has a new implementation of FFT. 
As a result the code is about 30% faster.
- Previous version for [PyTorch 1.6.0](https://pytorch.org/) is avaiable at `FNO-torch.1.6`.

## Files
The code is in the form of simple scripts. Each script shall be stand-alone and directly runnable.

- `exemples_paper/fourier_1d_exemple.py` is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
- `exemples_paper/fourier_2d_exemple.py` is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
- `exemples_paper/fourier_2d_time_exemple.py` is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf), which uses a recurrent structure to propagates in time.
- `exemples_paper/fourier_3d_exemple.py` is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf), which takes the 2D spatial + 1D temporal equation directly as a 3D problem
- The lowrank methods are similar. These scripts are the Lowrank neural operators for the corresponding settings.
- `data_generation` are the conventional solvers we used to generate the datasets for the Burgers equation, Darcy flow, and Navier-Stokes equation.

## Datasets
We provide the Burgers equation, Darcy flow, and Navier-Stokes equation datasets we used in the paper. 
The data generation configuration can be found in the paper.
- [PDE datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

The datasets are given in the form of matlab file. They can be loaded with the scripts provided in utilities.py. 
Each data file is loaded as a tensor. The first index is the samples; the rest of indices are the discretization.
For example, 
- `Burgers_R10.mat` contains the dataset for the Burgers equation. It is of the shape [1000, 8192], 
meaning it has 1000 training samples on a grid of 8192.
- `NavierStokes_V1e-3_N5000_T50.mat` contains the dataset for the 2D Navier-Stokes equation. It is of the shape [5000, 64, 64, 50], 
meaning it has 5000 training samples on a grid of (64, 64) with 50 time steps.

We also provide the data generation scripts at `data_generation`.

## Models
Here are the pre-trained models. It can be evaluated using _eval.py_ or _super_resolution.py_.
- [models](https://drive.google.com/drive/folders/1swLA6yKR1f3PKdYSKhLqK4zfNjS9pt_U?usp=sharing)

## Citations

```
@misc{li2020fourier,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{li2020neural,
      title={Neural Operator: Graph Kernel Network for Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2003.03485},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Future work

We are currently adding some new work to the repo :

- [ ] Factorized Fourier Neural Operator
- [ ] Conditioned Fourier Neural Operator


