import os
import time
import csv

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F

from fenics import *
from dolfin import *
from fenicstools.Interpolation import interpolate_nonmatching_mesh
from model.teecnet import TEECNet


class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class BurgersSolver():
    """
    A FEniCS implemented solver for Burgers' equation
    """
    def __init__(self, mesh, mesh_high, dt, nu, initial_condition, boundary_condition, integrated=False):
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
        self.integrated = integrated
        if not integrated:
            self.start_time = time.time()
            self.logger = Logger('runs/log_direct.txt', ['time_step', 'solution_time'])
        self.mesh = mesh
        if self.integrated:
            self.mesh_high = mesh_high
            self.V_high = FunctionSpace(self.mesh_high, 'P', 2)
        self.dt = dt
        self.nu = nu

        self.V = FunctionSpace(self.mesh, 'P', 2)

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
                # Apply the Neumann boundary condition as an appending term to the weak form     
                self.L += Constant(bc_value) * self.v * ds(bc_location)       
            else:
                raise ValueError('Boundary condition type must be either Dirichlet or Neumann')
        return bc
    
    def solve(self, **kwargs):
        """
        Solve the Burger's equation at the next time step
        """
        # Compute solution
        solve(self.a == self.L, self.u_n, self.bc)
        self.u_.assign(self.u_n)
        elapsed_time = time.time() - self.start_time
        if not self.integrated:
            self.logger.log({'time_step': kwargs['i'], 'solution_time': elapsed_time})

    def save_solution(self):
        """
        Save the solution on the high resolution mesh
        """
        file = XDMFFile('runs/solution_direct.xdmf')
        file.write(self.u_)
        file.close()


class IntergratedBurgersSolver():
    """
    An integrated PyTorch + FEniCS solver for Burgers' equation. The solver performs the following steps:
        1. Solve the Burgers' equation on a low resolution mesh using FEniCS
        2. Interpolate the solution onto a high resolution mesh using FEniCS
        3. Correct the interpolated solution with TEECNet and output the corrected solution as the solution on the high resolution mesh
        4. Update the solution on the low resolution mesh with the corrected solution as the initial condition for the next time step
    """
    def __init__(self, model_dir, mesh, mesh_high, dt, T, nu, initial_condition, boundary_condition):
        """
        Args:
            model_dir (str): Path to the directory of the trained TEECNet model
            mesh (dolfin.cpp.mesh.Mesh): The low resolution mesh on which the Burgers' equation is solved
            mesh_high (dolfin.cpp.mesh.Mesh): The high resolution mesh on which the solution is interpolated
            dt (float): The time step size
            T (float): The final time
            nu (float): The viscosity coefficient
            initial_condition (dolfin.function.function.Function): The initial condition
            boundary_condition (dolfin.function.function.Function): The boundary condition
        """
        self.start_time = time.time()
        self.logger = Logger('runs/log_integrated.txt', ['time_step', 'solution_time'])
        self.model_dir = model_dir
        self.model = TEECNet(1, 16, 1, num_layers=3, retrieve_weight=False)
        self.model.load_state_dict(torch.load(self.model_dir))
        self.model.eval()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.model.to(self.device)

        self.solver = BurgersSolver(mesh, mesh_high, dt, nu, initial_condition, boundary_condition, integrated=True)
        self.mesh = mesh
        self.mesh_high = mesh_high

        self.num_time_steps = int(T / dt)

        # construct pyg Data object for TEECNet
        self.u_high = interpolate()
        x = self.u_high.compute_vertex_values(self.mesh_high).unsqueeze(0)
        x = torch.from_numpy(x)
        pos = self.mesh_high.coordinates()
        edge_index = torch.zeros((2*self.mesh_high.num_edges(), 2), dtype=torch.long)
        edge_length = torch.zeros((2*self.mesh_high.num_edges()), dtype=torch.float)
        for i, edge in enumerate(self.mesh_high.edges()):
            edge_index[2*i, :] = edge.entities(0)
            edge_index[2*i+1, :] = torch.flipud(edge.entities(0))
            edge_length[2*i] = edge.length()
            edge_length[2*i+1] = edge.length()
        edge_index = edge_index.t().contiguous()
        edge_length = edge_length.unsqueeze(1)

        edge_attr = torch.cat((edge_length, torch.from_numpy(pos[edge_index[0]]), torch.from_numpy(pos[edge_index[1]])), dim=1)
        self.graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos).to(self.device)

    def solve(self):
        """
        Solve the Burgers' equation at the next time step
        """
        for i in range(self.num_time_steps):
            print('Time step: %d' % i)
            self.solver.solve()
            self.u_high = self.interpolate()
            self.correct()
            self.update()
            self.save_solution()
            # compute the elapsed time
            elapsed_time = time.time() - self.start_time
            self.logger.log({'time_step': i, 'solution_time': elapsed_time})

    def interpolate(self): 
        """
        Interpolate the solution onto a high resolution mesh
        """
        u_h = interpolate_nonmatching_mesh(self.solver.u_, self.solver.V_high)
        return u_h

    def correct(self):
        """
        Correct the interpolated solution with TEECNet
        """
        x = self.u_high.compute_vertex_values(self.mesh_high).unsqueeze(0)
        x = torch.from_numpy(x)
        # construct pyg Data object for TEECNet
        self.graph.x = x.to(self.device)
        with torch.no_grad():
            pred = self.model(self.graph).cpu()
        self.u_high.vector()[:] = pred.squeeze().numpy()
        self.u_high.vector().apply('insert')

    def update(self):
        """
        Update the solution on the low resolution mesh with the corrected solution as the initial condition for the next time step
        """
        self.solver.u_.assign(interpolate_nonmatching_mesh(self.u_high, self.solver.V))

    def save_solution(self):
        """
        Save the solution on the high resolution mesh
        """
        file = XDMFFile('runs/solution_integrated.xdmf')
        file.write(self.u_high)
        file.close()
