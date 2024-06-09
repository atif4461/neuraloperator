"""
@author: Mohammad Atif, CSI BNL
Saving visualizations as PNGs and GIFs
"""

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.animation as animation

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_vorticity_png(inputs, ground, pred, sample_index, file_name):

    with torch.no_grad():
        fig, axs = plt.subplots(figsize=(40, 15))
        
        vmax =  10#max(torch.max(inputs), torch.max(ground), torch.max(pred) )
        vmin = -10#min(torch.min(inputs), torch.min(ground), torch.min(pred) )
    
        for a in range(0,10):
            ax2 = fig.add_subplot(4,10,a+1)
            im=ax2.imshow(inputs[sample_index,:,:,a], vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
            plt.axis('off')
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        for a in range(0,10):
            ax2 = fig.add_subplot(4,10,a+11)
            im=ax2.imshow(ground[sample_index,:,:,a], vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
            plt.axis('off')
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        for a in range(0,10):
            ax2 = fig.add_subplot(4,10,a+21)
            im=ax2.imshow(pred[sample_index,:,:,a], vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
            plt.axis('off')
            ax2.set_xticks([])
            ax2.set_yticks([])
     
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.86, 0.15, 0.01, 0.7])
        cbar_ax.tick_params(labelsize=20)
        fig.colorbar(im,cax=cbar_ax)
     
        norm=plt.Normalize(-5,5)
        cmap1 = clr.LinearSegmentedColormap.from_list("", ["orange","white","green"])
        
        for a in range(0,10):
            ax2 = fig.add_subplot(4,10,a+31)
            im=ax2.imshow(pred[sample_index,:,:,a]-ground[sample_index,:,:,a], cmap=cmap1, norm=norm, interpolation='nearest')
            plt.axis('off')
            ax2.set_xticks([])
            ax2.set_yticks([])
    
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.89, 0.15, 0.01, 0.7])
        cbar_ax.tick_params(labelsize=20)
        fig.colorbar(im,cax=cbar_ax)
    
        l1 = get_rel_l1_error ( ground, pred, 10, sample_index )
        l2 = get_rel_l2_error ( ground, pred, 10, sample_index )
    
        plt.text(-75,   -6, str(l2[0].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        plt.text(-67.5, -6, str(l2[1].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        plt.text(-60,   -6, str(l2[2].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        plt.text(-53,   -6, str(l2[3].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        plt.text(-45.5, -6, str(l2[4].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        plt.text(-38,   -6, str(l2[5].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        plt.text(-30,   -6, str(l2[6].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        plt.text(-23,   -6, str(l2[7].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        plt.text(-16,   -6, str(l2[8].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        plt.text(-8.5,  -6, str(l2[9].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
        fig.savefig(file_name+str(sample_index)+'.png')
        plt.close()

def save_vorticity_gif(inputs, ground, pred, sample_index, file_name):

    with torch.no_grad():
        fig = plt.figure(figsize=(5, 10))    
        ax0 = fig.add_subplot(3,1,1)
        ax1 = fig.add_subplot(3,1,2)
        ax2 = fig.add_subplot(3,1,3)
    
        vmax =  10#max(torch.max(inputs), torch.max(ground), torch.max(pred) )
        vmin = -10#min(torch.min(inputs), torch.min(ground), torch.min(pred) )
        norm=plt.Normalize(-5,5)
        cmap1 = clr.LinearSegmentedColormap.from_list("", ["orange","white","green"])
    
        l1 = get_rel_l1_error ( ground, pred, 10, sample_index )
        l2 = get_rel_l2_error ( ground, pred, 10, sample_index )
    
        ims = []
    
        for a in range(0,10):
            im0 = ax0.imshow ( ground[sample_index,:,:,a], vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
            ax0.axis('off')
            im1 = ax1.imshow ( pred[sample_index,:,:,a], vmin=vmin, vmax=vmax, cmap='bwr', interpolation='nearest')
            ax1.axis('off')
            im2 = ax2.imshow ( pred[sample_index,:,:,a]-ground[sample_index,:,:,a], cmap=cmap1, norm=norm, interpolation='nearest')
            ax2.axis('off')
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
    
            im3 = plt.text(0.15, -1, str(l2[a].numpy()), bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=20)
            ims.append([im0, im1, im2, im3])
        
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.07, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=15)
        fig.colorbar(im2,cax=cbar_ax)
     
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=15)
        fig.colorbar(im0,cax=cbar_ax)
    
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat=True, repeat_delay=100000)
    
        ani.save(file_name+str(sample_index)+'.gif', writer='Pillow')
        plt.close()


