import os

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F

from fenics import *
from dolfin import *
from fenicstools.Interpolation import interpolate_nonmatching_mesh
from teecnet import TEECNet


class BurgersSolver():
    """
    A FEniCS implemented solver for Burgers' equation
    """
    def __init__(self, mesh, mesh_high, dt, nu, initial_condition, boundary_condition):
        """
        Args:
            mesh (dolfin.cpp.mesh.Mesh): The mesh on which the Burgers' equation is solved
            mesh_high (dolfin.cpp.mesh.Mesh): The high resolution mesh on which the solution is interpolated
            dt (float): The time step size
            nu (float): The viscosity coefficient
            initial_condition (dolfin.function.function.Function): The initial condition
            boundary_condition (dolfin.function.function.Function): The boundary condition
            source_term (dolfin.function.function.Function): The source term
        """
        self.mesh = mesh
        self.mesh_high = mesh_high
        self.dt = dt
        self.nu = nu

        self.V = FunctionSpace(self.mesh, 'P', 2)
        self.V_high = FunctionSpace(self.mesh_high, 'P', 2)
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.u_init = self.set_initial_condition(initial_condition)
        self.u_ = interpolate(self.u_init, self.V)

        self.u_n = Function(self.V)

        # Define variational problem
        F = inner((self.u - self.u_) / dt, self.v) * dx - dot(self.v, dot(self.u_, grad(self.u))) * dx \
        + self.nu * inner(grad(self.u), grad(self.v)) * dx

        # Create bilinear and linear forms
        self.a = lhs(F)
        self.L = rhs(F)

        # Define boundary condition
        self.bc = self.set_boundary_condition(boundary_condition)

    def set_initial_condition(self, initial_condition):
        """
        Set the initial condition either as a 'Expression' or by interpolating a 'Function' from a higher resolution mesh
        Args:
            initial_condition (dolfin.function.function.Function) / (dolfin.function.expression.Expression): The initial condition
        """
        if isinstance(initial_condition, Expression):
            u_init = interpolate(initial_condition, self.V)
        elif isinstance(initial_condition, Function):
            u_init = initial_condition
        else:
            raise TypeError('Initial condition must be either an Expression or a Function')
        return u_init

    def set_boundary_condition(self, boundary_condition):
        """
        Set the boundary condition as Dirichlet or Neumann
        Args:
            boundary_condition: Python list of boundary conditions
                example: [['Dirichlet', 'on_boundary', '0'], ['Neumann', 'on_boundary', '0']]
        """
        bc = []
        for bc_type, bc_location, bc_value in boundary_condition:
            if bc_type == 'Dirichlet':
                bc.append(DirichletBC(self.V, Constant(bc_value), bc_location))
            elif bc_type == 'Neumann':
                bc.append(DirichletBC(self.V, Constant(bc_value), bc_location))
            else:
                raise ValueError('Boundary condition type must be either Dirichlet or Neumann')
        return bc


class IntergratedBurgersSolver():
    """
    An integrated PyTorch + FEniCS solver for Burgers' equation. The solver performs the following steps:
        1. Solve the Burgers' equation on a low resolution mesh using FEniCS
        2. Interpolate the solution onto a high resolution mesh using FEniCS
        3. Correct the interpolated solution with TEECNet and output the corrected solution as the solution on the high resolution mesh
        4. Update the solution on the low resolution mesh with the corrected solution as the initial condition for the next time step
    """
    def __init__(self, model_dir):
        """
        Args:
            model_dir (str): Path to the directory of the trained TEECNet model
        """
        self.model_dir = model_dir
        self.model = TEECNet(1, 16, 1, num_layers=3, retrieve_weight=False)
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'model.pt')))
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        

        