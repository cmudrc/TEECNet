# CFDError

The codebase for sequencial training and evaluation of the discretization error estimation model.

## Overview

Discretization errors pose a unique challenge in the super-resolution process of physical simulations. Such errors are often difficult to quantify and can be highly non-linear. In this work, we propose a novel approach to estimate the discretization error in a super-resolution process. We use a graph neural network (GNN) to learn the mapping between the low-resolution (LR) and high-resolution (HR) simulation results. The GNN is trained to minimize the mean squared error (MSE) between the predicted and ground truth HR properties. The discretization error is then estimated as the difference between the predicted and ground truth HR images. We demonstrate the effectiveness of our approach on one numerical scheme and two different physical simulations: estimating surface area of an ellipse, fluid flow and heat transfer. We show that our approach can accurately estimate the discretization error and can be used to improve the super-resolution process.