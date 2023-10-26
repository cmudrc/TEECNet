import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

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
            self.save_file = XDMFFile('runs/solution_direct.xdmf')
        self.mesh = mesh
        if self.integrated:
            self.mesh_high = mesh_high
            self.V_high = VectorFunctionSpace(self.mesh_high, 'P', 1)
        self.dt = dt
        self.nu = nu

        self.t = 0 # current time

        self.V = VectorFunctionSpace(self.mesh, 'P', 1)

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.u_ = self.set_initial_condition(initial_condition)

        self.u_n = Function(self.V)

        # Define variational problem
        F = inner((self.u - self.u_) / dt, self.v)*dx - dot(self.v, dot(self.u_, grad(self.u)))*dx \
        + self.nu * inner(grad(self.u), grad(self.v))*dx

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
                # self.L += Constant(bc_value) * self.v * ds(bc_location)  
                pass     
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
        
        if not self.integrated:
            elapsed_time = time.time() - self.start_time
            self.t += self.dt
            self.logger.log({'time_step': kwargs['i'], 'solution_time': elapsed_time})
            self.save_solution()

    def save_solution(self):
        """
        Save the solution on the high resolution mesh
        """
        self.save_file.write(self.u_, self.t)


class IntergratedBurgersSolver():
    """
    An integrated PyG + FEniCS solver for Burgers' equation. The solver performs the following steps:
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
        self.logger = Logger('runs/log_integrated.txt', ['time_step', 'solution_time', 'solve_time', 'interpolation_time', 'correction_time', 'update_time'])
        self.model_dir = model_dir
        self.model = TEECNet(2, 16, 2, num_layers=3, retrieve_weight=False)
        self.model.load_state_dict(torch.load(self.model_dir, map_location='cpu'))
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model.to(self.device)

        self.solver = BurgersSolver(mesh, mesh_high, dt, nu, initial_condition, boundary_condition, integrated=True)
        self.mesh = mesh
        self.mesh_high = mesh_high

        self.num_time_steps = int(T / dt)
        self.t = 0 # current time

        # construct pyg Data object for TEECNet
        self.u_high = self.interpolate()
        x = self.u_high.vector()[:]
        u_x = x[::2].astype(np.float32)
        u_y = x[1::2].astype(np.float32)
        x = np.stack((u_x, u_y), axis=1)
        x = torch.from_numpy(x)
        pos = np.array(self.mesh_high.coordinates(), dtype=np.float32)
        edge_index = torch.zeros((2*self.mesh_high.num_edges(), 2), dtype=torch.int64)
        edge_length = torch.zeros((2*self.mesh_high.num_edges()), dtype=torch.float32)
        for i, edge in enumerate(edges(self.mesh_high)):
            edge_index[2*i, :] = torch.from_numpy(np.array(edge.entities(0), dtype=np.int64))
            edge_index[2*i+1, :] = torch.flipud(torch.from_numpy(np.array(edge.entities(0), dtype=np.int64)))
            edge_length[2*i] = torch.from_numpy(np.array(edge.length(), dtype=np.float32))
            edge_length[2*i+1] = torch.from_numpy(np.array(edge.length(), dtype=np.float32))
        edge_index = edge_index.t().contiguous()
        edge_length = edge_length.unsqueeze(1)

        edge_attr = torch.cat((edge_length, torch.from_numpy(pos[edge_index[0]]), torch.from_numpy(pos[edge_index[1]])), dim=1)
        self.graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos).to(self.device)

        self.save_file = XDMFFile('runs/solution_integrated.xdmf')
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

    def solve(self):
        """
        Solve the Burgers' equation at the next time step
        """
        for i in range(self.num_time_steps):
            self.t += self.solver.dt
            time_1 = time.time()
            self.solver.solve()
            time_2 = time.time()
            time_solve = time_2 - time_1
            self.u_high = self.interpolate()
            time_3 = time.time()
            time_interpolate = time_3 - time_2
            self.correct()
            time_4 = time.time()
            time_correct = time_4 - time_3
            self.update()
            time_5 = time.time()
            time_update = time_5 - time_4
            self.save_solution()
            # compute the elapsed time
            elapsed_time = time.time() - self.start_time
            self.logger.log({'time_step': i, 'solution_time': elapsed_time, 'solve_time': time_solve, 'interpolation_time': time_interpolate, 'correction_time': time_correct, 'update_time': time_update})

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
        x = np.array(self.u_high.vector(), dtype=np.float32)
        # u_x is the 0, 2, 4, ... elements of x
        u_x = x[::2]
        # u_y is the 1, 3, 5, ... elements of x
        u_y = x[1::2]
        x = np.stack((u_x, u_y), axis=1)
        x = torch.from_numpy(x)
        # construct pyg Data object for TEECNet
        self.graph.x = x.to(self.device)
        with torch.no_grad():
            x, edge_index, edge_attr = self.graph.x, self.graph.edge_index, self.graph.edge_attr
            pred = self.model(x, edge_index, edge_attr).cpu()
        # insert pred x and y components into the solution vector. x component goes into 0, 2, 4, ... elements of the vector, y component goes into 1, 3, 5, ... elements of the vector
        pred_insert = np.zeros((2*len(pred)))
        pred_insert[::2] = pred[:, 0].numpy()
        pred_insert[1::2] = pred[:, 1].numpy()
        self.u_high.vector()[:] = pred_insert
        self.u_high.vector().apply('insert')

    def update(self):
        """
        Update the solution on the low resolution mesh with the corrected solution as the initial condition for the next time step
        """
        self.solver.u_.assign(interpolate_nonmatching_mesh(self.u_high, self.solver.V))

    def save_solution(self):
        """
        Save the solution on the high resolution mesh at the current time step
        """
        self.save_file.write(self.u_high, self.t)
        u = self.u_high.vector()[:]
        u_x = u[::2]
        u_y = u[1::2]
        u_mag = np.sqrt(u_x**2 + u_y**2)

        self.ax.cla()
        self.ax.set_title('t = {}'.format(self.t))
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.set_aspect('equal')
        self.ax.tricontourf(self.mesh_high.coordinates()[:, 0], self.mesh_high.coordinates()[:, 1], self.mesh_high.cells(), u_mag, cmap='jet', levels=100)
        self.fig.savefig('runs/figures/t_{}.png'.format(self.t), dpi=300)
