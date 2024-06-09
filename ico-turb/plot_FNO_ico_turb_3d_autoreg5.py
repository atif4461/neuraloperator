"""
Training a TFNO on Double Periodic Shear Layer
==============================================

In this example, we demonstrate how to use the small 3D DPSL example
to train a Tensorized Fourier-Neural Operator
"""

# %%
# 

import torch
import matplotlib.pyplot as plt
import h5py
import sys
import numpy as np
import plotly.graph_objects as go

from neuralop.models import TFNO
from neuralop.models import FNO
from neuralop.datasets import Normalizer
from neuralop.datasets import UnitGaussianNormalizer
from ico_turb_3d import load_ico_turb_autoreg_3d
from neuralop.utils import count_model_params
from neuralop import Trainer
from neuralop import LpLoss, H1Loss
from utils import get_rel_l1_error, get_rel_l2_error, energy_spectrum 

import inspect

device = torch.device('cuda')

ntrain = 1000
ntest = 10
resolution = 256
batch_size = 8
T_in = 10
T = 5

TRAIN_PATH = '/work/atif/datasets/dpsl_Re1e4_dx_256/dpsl_data_N5000_T201_ux_uy_vort.h5'
ftrain = h5py.File(TRAIN_PATH, 'r')
print(ftrain.keys())

train_a = torch.tensor(ftrain['vortz'][0:ntrain,:,:,:])       

print("train_a.shape", train_a.shape)



# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_ico_turb_autoreg_3d(
        n_train=ntrain, batch_size=batch_size, T_in=T_in, T=T, 
        test_resolutions=[resolution], n_tests=[ntest],
        test_batch_sizes=[batch_size],
        positional_encoding=False
)
data_processor = data_processor.to(device)
print(data_processor)

# Hyperparameters
width = 8
scheduler_step = 100
scheduler_gamma = 0.5
learning_rate = 0.001
epochs = 501
nmodes = 32

# %%
# We create a tensorized FNO model
# I think channel_dim is for number of inputs --  1 for vorticity, 2 for velocity in 2D and 3 for both in 3D
# or is it like each T_in is a separate channel?
model = FNO(n_modes=(nmodes, nmodes), hidden_channels=width, in_channels=T_in, out_channels=T, T_in=1, T_out=1)
model = model.to(device)


n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate, 
                                weight_decay=1e-4) # sensitive parameter
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()



# %% 
# Create the trainer
trainer = Trainer(model=model, n_epochs=epochs,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)


print('\n### TRAINING FINISHED ###\n')

torch.save(model, f'./model_autoreg5_{ntrain}_{batch_size}_{width}_{scheduler_step}_{scheduler_gamma}_{learning_rate}_{nmodes}_final')





## # # Inference # # #
#
##model = torch.load('/work/atif/neuraloperator/ico-turb/test_channels')
### Port the model to GPU for inference
#model = model.cuda()
#
#TEST_PATH = '/work/atif/datasets/dpsl_Re1e4_dx_256/dpsl_data_N5000_T201_ux_uy_vort.h5'
#ftest = h5py.File(TEST_PATH, 'r')
#
#x_test = torch.tensor(ftest['vortz'][0:3, 0:10,  :, :]).type(torch.float32)
#y_test = torch.tensor(ftest['vortz'][0:3, 10:20, :, :]).type(torch.float32)
#
#for index in range(0,3):
#    
#    mean, std = torch.mean(x_test[index:index+1, 0:1, :, :]), torch.std(x_test[index:index+1, 0:1, :, :])
#    x_test = (x_test - mean)/(std + 1e-6) 
#    y_test = (y_test - mean)/(std + 1e-6) 
#     
#    with torch.no_grad():
#        # Model prediction
#        out = model(x_test.cuda())
#    
#    print("x y out shapes = ", x_test.shape, y_test.shape, out.shape)
#    print('\n### INFERENCE FINISHED ###\n')
#    print(f'input  limits are ({torch.min(x_test)}, {torch.max(x_test)})')
#    print(f'output limits are ({torch.min(y_test)}, {torch.max(y_test)})')
#    print(f'inferc limits are ({torch.min(out)}, {torch.max(out)})')
#    
#    # Plotting a sample vorticity heat map
#    
#    vmax = (torch.max(x_test))
#    vmin = (torch.min(x_test))
#     
#    fig, axs = plt.subplots(figsize=(150, 40))
#    
#    for a in range (0,10):
#        ax2 = fig.add_subplot(3,10,a+1)
#        im  = ax2.imshow(x_test[index,a,:,:].cpu(), vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
#        plt.axis('off')
#        ax2.set_xticks([])
#        ax2.set_yticks([])
#    
#        ax2 = fig.add_subplot(3,10,a+11)
#        im  = ax2.imshow(y_test[index,a,:,:].cpu(), vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
#        plt.axis('off')
#        ax2.set_xticks([])
#        ax2.set_yticks([])
#    
#        ax2 = fig.add_subplot(3,10,a+21)
#        im  = ax2.imshow(out[index,a,:,:].cpu(), vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
#        plt.axis('off')
#        ax2.set_xticks([])
#        ax2.set_yticks([])
#    
#    
#    fig.subplots_adjust(right=0.85)
#    cbar_ax = fig.add_axes([0.86, 0.15, 0.01, 0.7])
#    cbar_ax.tick_params(labelsize=20)
#    fig.colorbar(im,cax=cbar_ax)
#    
#    fig.savefig(f'test_{index}_3d_channels.png')
#    plt.close()

# Compute errors for 500 test samples
model = torch.load(f'./model_autoreg5_{ntrain}_{batch_size}_{width}_{scheduler_step}_{scheduler_gamma}_{learning_rate}_{nmodes}_final')
## Port the model to GPU for inference
model = model.cuda()

TEST_PATH = '/work/atif/datasets/dpsl_Re1e4_dx_256/dpsl_data_N5000_T201_ux_uy_vort.h5'
ftest = h5py.File(TEST_PATH, 'r')
l2er = np.zeros(10)
corr = np.zeros(10)
tarr = range(0,10)
flat_tensor1_2 = np.zeros(10)
flat_tensor2_2 = np.zeros(10)
x_test = torch.empty(10,256,256)
y_test = torch.empty(1,256,256)

for batch in range(0,5):
    x_test_all = torch.tensor(ftest['vortz'][batch*100:(batch+1)*100, 0:10,  :, :]).type(torch.float32)
    y_test_all = torch.tensor(ftest['vortz'][batch*100:(batch+1)*100, 10:20, :, :]).type(torch.float32)

    for index in range(0,100):

        x_sample = x_test_all[index, 0:10,  :, :]
        y_sample = y_test_all[index, 0:10, :, :]
        mean, std = torch.mean(x_sample[0, :, :]), torch.std(x_sample[0, :, :])
            
        x_test[0, :, :] = x_sample[0, :, :]
        x_test[1, :, :] = x_sample[1, :, :]
        x_test[2, :, :] = x_sample[2, :, :]
        x_test[3, :, :] = x_sample[3, :, :]
        x_test[4, :, :] = x_sample[4, :, :]
        x_test[5, :, :] = x_sample[5, :, :]
        x_test[6, :, :] = x_sample[6, :, :]
        x_test[7, :, :] = x_sample[7, :, :]
        x_test[8, :, :] = x_sample[8, :, :]
        x_test[9, :, :] = x_sample[9, :, :]

        x_test = (x_test - mean)/(std + 1e-6) 
        y_test = (y_test - mean)/(std + 1e-6) 
        
        with torch.no_grad():
            # Model prediction
            out = model(x_test.unsqueeze(0).cuda())
       
        out1_unnrm = out * std + mean;

        print('\n### INFERENCE FINISHED ###\n')
        print(f'input  limits are ({torch.min(x_test)}, {torch.max(x_test)})')
        print(f'output limits are ({torch.min(y_test)}, {torch.max(y_test)})')
        print(f'inferc limits are ({torch.min(out)}, {torch.max(out)})')
        print("x y out shapes = ", x_test.shape, y_test.shape, out.shape)

        x_test[0, :, :] = x_sample[5, :, :]
        x_test[1, :, :] = x_sample[6, :, :]
        x_test[2, :, :] = x_sample[7, :, :]
        x_test[3, :, :] = x_sample[8, :, :]
        x_test[4, :, :] = x_sample[9, :, :]
        x_test[5, :, :] = out1_unnrm[0, 0, :, :] 
        x_test[6, :, :] = out1_unnrm[0, 1, :, :]
        x_test[7, :, :] = out1_unnrm[0, 2, :, :]
        x_test[8, :, :] = out1_unnrm[0, 3, :, :]
        x_test[9, :, :] = out1_unnrm[0, 4, :, :]
        
        x_test = (x_test - mean)/(std + 1e-6) 

        with torch.no_grad():
            # Model prediction
            out = model(x_test.unsqueeze(0).cuda())
        
        out2_unnrm = out * std + mean;
        
        print('\n### INFERENCE FINISHED ###\n')
        print(f'input  limits are ({torch.min(x_test)}, {torch.max(x_test)})')
        print(f'output limits are ({torch.min(y_test)}, {torch.max(y_test)})')
        print(f'inferc limits are ({torch.min(out)}, {torch.max(out)})')
        print("x y out shapes = ", x_test.shape, y_test.shape, out.shape)

        x_test[0, :, :] = out1_unnrm[0, 0, :, :] 
        x_test[1, :, :] = out1_unnrm[0, 1, :, :] 
        x_test[2, :, :] = out1_unnrm[0, 2, :, :] 
        x_test[3, :, :] = out1_unnrm[0, 3, :, :] 
        x_test[4, :, :] = out1_unnrm[0, 4, :, :] 
        x_test[5, :, :] = out2_unnrm[0, 0, :, :] 
        x_test[6, :, :] = out2_unnrm[0, 1, :, :] 
        x_test[7, :, :] = out2_unnrm[0, 2, :, :] 
        x_test[8, :, :] = out2_unnrm[0, 3, :, :] 
        x_test[9, :, :] = out2_unnrm[0, 4, :, :] 
 
        out = (x_test.unsqueeze(0) - mean)/(std + 1e-6) 
        x_test = (x_sample - mean)/(std + 1e-6) 
        y_test = (y_sample - mean)/(std + 1e-6) 

        print('\n### INFERENCE FINISHED ###\n')
        print(f'input  limits are ({torch.min(x_test)}, {torch.max(x_test)})')
        print(f'output limits are ({torch.min(y_test)}, {torch.max(y_test)})')
        print(f'inferc limits are ({torch.min(out)}, {torch.max(out)})')
        print("x y out shapes = ", x_test.shape, y_test.shape, out.shape)

        out = out.cpu()

        for t in range (0,10):
            l2er[t] += get_rel_l2_error( y_test[t,:,:], out[0,t,:,:] )
            flat_tensor1 = torch.flatten(y_test[t,:,:])
            flat_tensor2 = torch.flatten(out[0,t,:,:])
            inner_product = np.inner( flat_tensor1, flat_tensor2)
            corr[t] += inner_product / np.sqrt( np.inner(flat_tensor1,flat_tensor1) * np.inner(flat_tensor2,flat_tensor2) )
            flat_tensor1_2[t] += np.linalg.norm(y_test[t,:,:],2)
            flat_tensor2_2[t] += np.linalg.norm(out[0,t,:,:],2)


np.savetxt( f'corr_l2_3d_autoreg5_width{width}.txt', np.column_stack( [tarr, corr/500.0, l2er/500.0,  flat_tensor1_2/500.0, flat_tensor2_2/500.0] ) )
