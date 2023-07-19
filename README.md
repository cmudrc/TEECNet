# TEECNet

The codebase for sequencial training and evaluation of the discretization error estimation model.

## Overview

Discretization errors pose a unique challenge in the super-resolution process of physical simulations. Such errors are often difficult to quantify and can be highly non-linear. In this work, we propose a novel approach to estimate the discretization error in a super-resolution process. We use a graph neural network (GNN) to learn the mapping between the low-resolution (LR) and high-resolution (HR) simulation results. The GNN is trained to minimize the mean squared error (MSE) between the predicted and ground truth HR properties. The discretization error is then estimated as the difference between the predicted and ground truth HR images. We demonstrate the effectiveness of our approach on one numerical scheme and two different physical simulations: estimating surface area of an ellipse, fluid flow and heat transfer. We show that our approach can accurately estimate the discretization error and can be used to improve the super-resolution process.

## Quick Start

### Installation

1. Clone this repository.
```bash
git clone https://github.com/cmudrc/CFDError.git
```

2. Install the dependencies.
```bash
pip install -r requirements.txt
```

### Training
We provide several typical physics settings in this project. For example, to perform training on estimating the discretization error of the surface area of an ellipse, run the following command:
```bash
python test_case_ellipse.py
```
The supported test cases are:
- `test_case_ellipse.py`: surface area of an ellipse
- `test_case_heat.py`: 2-D heat transfer on an unit square domain
- `test_case_burger.py`: 2-D Burger's equation on an unit square domain

### Evaluation
To be completed.

