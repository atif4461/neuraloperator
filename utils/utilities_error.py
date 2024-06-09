"""
@author: Mohammad Atif
Calculations for relative L1, L2 errors
# Error calculation
https://arxiv.org/pdf/2402.17185v1
https://arxiv.org/pdf/2401.08886v1
https://arxiv.org/pdf/2206.02016.pdf
"""

import torch
import numpy as np
import pyfftw

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.animation as animation

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_rel_l1_error ( ground, pred ):

    with torch.no_grad():
        Nx = ground.shape[0]
        Ny = ground.shape[1]
    
        #pred = pred.to(device) 
        #ground = ground.to(device) 
    
        loss  = 0.0 
        denom = 0.0 
    
        for i in range (0, Nx):
            for j in range (0, Ny):
                loss  = torch.abs ( ground[i, j] - pred[i, j] )
                denom = torch.abs ( ground[i, j] )
    
        l1 = np.sqrt( loss/denom )
    
        return l1.cpu()

def get_rel_l2_error ( ground, pred ):

    with torch.no_grad():
        Nx = ground.shape[0]
        Ny = ground.shape[1]
    
        #pred = pred.to(device) 
        #ground = ground.to(device) 
        
        loss_t = torch.nn.functional.mse_loss ( pred[:,:], ground[:,:], reduction='sum' )
        norm_t = torch.linalg.norm ( ground[:,:] )
        
        l2 = np.sqrt( loss_t ) / norm_t

        #loss  = 0.0 
        #denom = 0.0 
        #for i in range (0, Nx):
        #    for j in range (0, Ny):
        #        loss += (pred[index, i, j, t] - ground[index, i, j, t])**2
        #        denom += (ground[index, i, j, t])**2
    
        #l2[t] += np.sqrt( loss[t]/denom[t] )
    
        return l2.cpu()

def energy_spectrum(w):
        
    nx = w.shape[0]
    ny = w.shape[1]
    
    '''
    Computation of energy spectrum and maximum wavenumber from vorticity field
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    w : vorticity field in physical spce (including periodic boundaries)
    
    Output
    ------
    en : energy spectrum computed from vorticity field
    n : maximum wavenumber
    '''
    
    epsilon = 1.0e-6

    kx = np.empty(nx)
    ky = np.empty(ny)
   
    dx = 1./nx
    dy = 1./ny

    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w[0:nx,0:ny]) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
                    
        en[k] = en[k]/ic
        
    return en, n
