import numpy as np
from matplotlib import pyplot as plt


import pyqg
from pyqg import diagnostic_tools as tools
import wandb

# initialize logging to monitor the run
wandb.init(project="pyqg")

L =  1000.e3     # length scale of box    [m]
Ld = 15.e3       # deformation scale      [m]
kd = 1./Ld       # deformation wavenumber [m^-1]
Nx = 64          # number of grid points

H1 = 500.        # layer 1 thickness  [m]
H2 = 1750.       # layer 2
H3 = 1750.       # layer 3

U1 = 0.05          # layer 1 zonal velocity [m/s]
U2 = 0.025         # layer 2
U3 = 0.00          # layer 3

rho1 = 1025.
rho2 = 1025.275
rho3 = 1025.640

rek = 1.e-7       # linear bottom drag coeff.  [s^-1]
f0  = 0.0001236812857687059 # coriolis param [s^-1]
beta = 1.2130692965249345e-11 # planetary vorticity gradient [m^-1 s^-1]

Ti = Ld/(abs(U1))  # estimate of most unstable e-folding time scale [s]
dt = Ti/200.   # time-step [s]
tmax = 500*Ti      # simulation time [s]

m = pyqg.LayeredModel(nx=Nx, nz=3, U = [U1,U2,U3],V = [0.,0.,0.],L=L,f=f0,beta=beta,
                         H = [H1,H2,H3], rho=[rho1,rho2,rho3],rek=rek,
                        dt=dt,tmax=tmax, twrite=10000, tavestart=Ti*200)

sig = 1.e-7
qi = sig*np.vstack([np.random.randn(m.nx,m.ny)[np.newaxis,],
                    np.random.randn(m.nx,m.ny)[np.newaxis,],
                    np.random.randn(m.nx,m.ny)[np.newaxis,]])
m.set_q(qi)

# run the model
m.run()

ds = m.to_dataset()

PV = ds.q + ds.Qy * ds.y
PV['x'] = ds.x/ds.attrs['pyqg:rd']; PV.x.attrs = {'long_name': r'$x/L_d$'}
PV['y'] = ds.y/ds.attrs['pyqg:rd']; PV.y.attrs = {'long_name': r'$y/L_d$'}

plt.figure(figsize=(18,4))

plt.subplot(131)
PV.sel(lev=1).plot(cmap='Spectral_r')

plt.subplot(132)
PV.sel(lev=2).plot(cmap='Spectral_r')

plt.subplot(133)
PV.sel(lev=3).plot(cmap='Spectral_r')

plt.savefig('test_ds.png')  