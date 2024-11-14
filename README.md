# TEECNet

The codebase for sequencial training and evaluation of the discretization error estimation model.

## Overview

Discretization errors pose a unique challenge in the super-resolution process of physical simulations. Such errors are often difficult to quantify and can be highly non-linear. In this work, we propose a novel approach to estimate the discretization error in a super-resolution process. We use a graph neural network (GNN) to learn the mapping between the low-resolution (LR) and high-resolution (HR) simulation results. The GNN is trained to minimize the mean squared error (MSE) between the predicted and ground truth HR properties. The discretization error is then estimated as the difference between the predicted and ground truth HR images. We demonstrate the effectiveness of our approach on one numerical scheme and two different physical simulations: estimating surface area of an ellipse, fluid flow and heat transfer. We show that our approach can accurately estimate the discretization error and can be used to improve the super-resolution process.

## Quick Start

### Installation

1. Clone this repository.
```bash
git clone https://github.com/cmudrc/TEECNet.git
```

2. Install the dependencies.
```bash
pip install -r requirements.txt
```

### Training
We provide several typical physics settings in this project, including heat transfer, Burgers' equation and Navier-Stokes equation. For downloading the training data, please refer to [data](). The user can also generate their own data samples from the [simulation code](https://github.com/WenzhuoXu/pdecal).

### Evaluation
We provide experimental codes for generating all the figures in the paper. The user can also use the trained model to estimate the discretization error in their own simulation. Four included experiments are as follows:
1. Multi-resolution generalization. In this experiment, we train the model on a specific resolution pair and test it on other resolution pairs. The user can adjust the `config.yaml` file to change the model to be trained, the training & resolution pairs to perform, and the physics scenarios. We provide the config to reproduce the results in the paper, and one example of running the experiment on TEECNet, on all resolution pairs and on the Burgers' equation would be:
```bash
python3 teecnet_exp_1_multi_resolution.py --config config/exp_1_burger.yaml
```
2. Model expressivenss. In this experiment, we train the model on a small number of sample-by-sample updates. After each model update we evaluate the performance on an unseen test set and plot the results. The user can adjust the `config.yaml` file to change the model to be trained, the training & resolution pairs to perform, and the physics scenarios. We provide the config to reproduce the results in the paper, and one example of running the experiment on TEECNet, on resolution pair [16, 64] and on the Heat transfer equation would be:
```bash
python3 teecnet_exp_2_expressiveness.py --config config/exp_2_heat.yaml
```
3. Geometrical invariance. In this experiment, we utilize our trained model from experiment 1 and test it on irregular meshes and geometries to verify the model performance. A new dataset is created specifically for this purpose, and can be found at [data]() as `heat_geometry_test`. Note that, only the heat transfer scenario is provided. The user can also generate their own set of geometry and simulation using the code provided [here](https://github.com/WenzhuoXu/pdecal). We provide the config to reproduce the results in the paper, and one example of running the experiment on TEECNet, on resolution pair [16, 64] and on the Heat transfer equation would be:
```bash
python3 teecnet_exp_3_geometrical.py --config config/exp_3_heat.yaml
```
4. An integrated PyTorch + FEniCS solver. In this experiment we provide an end-to-end use case demonstration for our model, and implemented an integrated numerical simulation + neural network solver with PyTorch and FEniCS for the Burgers' equation. At each time step the solver performs the following steps:
    1. Solve the Burgers' equation on a low resolution mesh using FEniCS
    2. Interpolate the solution onto a high resolution mesh using FEniCS
    3. Correct the interpolated solution with TEECNet and output the corrected solution as the solution on the high resolution mesh
    4. Update the solution on the low resolution mesh with the corrected solution as the initial condition for the next time step
We provide the config to reproduce the results in the paper, and one example of running the experiment on TEECNet, on resolution pair [10, 80] and on the Burgers' equation would be:
```bash
python3 teecnet_exp_4_integrated_solver.py --config config/exp_4_burger.yaml
```
