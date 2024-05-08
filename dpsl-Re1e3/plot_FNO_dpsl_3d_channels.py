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

from neuralop.models import TFNO
from neuralop.models import FNO
from neuralop.datasets import UnitGaussianNormalizer
from dpsl_3d import load_dpsl_3d_channels
from neuralop.utils import count_model_params
from neuralop import Trainer
from neuralop import LpLoss, H1Loss

import inspect

device = torch.device('cuda')

ntrain = 1024
ntest = 128
resolution=64
batch_size=32
T_in = 10
T = 10

# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_dpsl_3d_channels(
        n_train=ntrain, batch_size=batch_size, T_in=T_in, T=T, 
        test_resolutions=[resolution], n_tests=[ntest],
        test_batch_sizes=[batch_size],
        positional_encoding=False
)
data_processor = data_processor.to(device)
print(data_processor)

# Hyperparameters
width = 40
scheduler_step = 100
scheduler_gamma = 0.5
learning_rate = 0.001
epochs = 51

# %%
# We create a tensorized FNO model
# I think channel_dim is for number of inputs --  1 for vorticity, 2 for velocity in 2D and 3 for both in 3D
# or is it like each T_in is a separate channel?
model = FNO(n_modes=(8, 8), hidden_channels=width, in_channels=T_in, out_channels=T, T_in=1, T_out=1)
#model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
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

# Port the model to GPU for inference
model = model.cuda()

test_samples = test_loaders[resolution].dataset

for index in range(3):
    
    fig = plt.figure(figsize=(21, 7))
    
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    print(f'index {index}, xshape {x.shape}, yshape {y.shape}, {y[0,0].shape}, out shape {out.shape}')

    for t_index in range(T_in):
        ax = fig.add_subplot(3, T_in, t_index + 1)
        ax.imshow(x[t_index].cpu())
        ax.set_title('Input vorticity t='+str(t_index))
        plt.xticks([], [])
        plt.yticks([], [])

    for t_index in range(T):
        ax = fig.add_subplot(3, T, t_index + T_in + 1)
        ax.imshow(y[0,t_index].cpu().squeeze())
        ax.set_title('Output vorticity t='+str(t_index+T_in))
        plt.xticks([], [])
        plt.yticks([], [])

    for t_index in range(T):
        ax = fig.add_subplot(3, T, t_index + T_in + T + 1)
        ax.imshow(out[0,t_index].cpu().squeeze().detach().numpy())
        ax.set_title('Predicted vorticity t='+str(t_index+T_in))
        plt.xticks([], [])
        plt.yticks([], [])

    fig.suptitle('Inputs, ground-truth output and prediction for DPSL with 3D FNO', y=0.98)
    fig.subplots_adjust(wspace=0.0)
    plt.tight_layout()
    #fig.show()
    fig.savefig('dpsl_3d_channels_'+str(index)+'.png')

print('\n### INFERENCE FINISHED ###\n')
