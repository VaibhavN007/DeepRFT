# %% 

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from DeepRFT_MIMO import DeepRFT as myNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from get_parameter_number import get_parameter_number
import kornia
from torch.utils.tensorboard import SummaryWriter
import argparse

from customDataSet import CustomDataset

# %%
######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# %%
doc_dir = "./publaynet/"
result_dir = "./results/transfer"

result_dir   = os.path.join(result_dir)
original_dir = os.path.join(result_dir, "original")
blurred_dir  = os.path.join(result_dir, "blurred")
restored_dir = os.path.join(result_dir, "restored")
model_dir    = os.path.join(result_dir, "models")
log_dir      = os.path.join(result_dir, "logs")

utils.mkdir(result_dir)
utils.mkdir(original_dir)
utils.mkdir(blurred_dir)
utils.mkdir(restored_dir)
utils.mkdir(model_dir)
utils.mkdir(log_dir)

# %%
mode = 'Deblurring'

patch_size = 256    # patch size, for paper: [GoPro, HIDE, RealBlur]=256, [DPDD]=512
train_dir = blurred_dir
val_dir = original_dir

num_epochs = 2
batch_size = 16
val_epochs = 20

# %%
######### Model ###########
model_restoration = myNet()

# print number of model
get_parameter_number(model_restoration)

model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

# %%
weights_ = "./DeepRFT-MIMO-v1/DeepRFT/model_GoPro.pth"
utils.load_checkpoint(model_restoration, weights_)

# %%

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()

# %%
######### DataLoaders ###########
train_dataset = CustomDataset(doc_dir, output_dim=(256,256), filter=False, type="train")
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
len_dataset = len(train_dataset)

val_dataset = CustomDataset(doc_dir, output_dim=(256,256), filter=False, type="val")
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
len_valdataset = len(val_dataset)
# %%
start_epoch = 1

num_samples_per_epoch = 1000
num_val_samples = 400

data_epochs = len_dataset//num_samples_per_epoch
actual_epochs = 10

num_epochs = actual_epochs*data_epochs

num_tb_samples = 100
tb_epochs = len_dataset//num_tb_samples
# %%
new_lr = 2e-4
optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
# %%
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs-warmup_epochs, eta_min=1e-6)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()
# %%
best_psnr = 0
best_epoch = 0
writer = SummaryWriter(log_dir)
iter = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        input_  = data[0].cuda()
        target_ = data[1].cuda()

        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored = model_restoration(input_)

        loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(restored[2], target[2])
        loss_char = criterion_char(restored[0], target[0]) + criterion_char(restored[1], target[1]) + criterion_char(restored[2], target[2])
        loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(restored[1], target[1]) + criterion_edge(restored[2], target[2])
        loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge
        
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        if i%num_tb_samples == 0:
            writer.add_scalar("data/training_loss", loss.val, (epoch-1) * tb_epochs + (i//num_tb_samples))
        
        # iter += 1
        # writer.add_scalar('loss/fft_loss', loss_fft, iter)
        # writer.add_scalar('loss/char_loss', loss_char, iter)
        # writer.add_scalar('loss/edge_loss', loss_edge, iter)
        # writer.add_scalar('loss/iter_loss', loss, iter)

        #### Evaluation ####
        if i%num_samples_per_epoch == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_loader), 0):
                input_ = data_val[0].cuda()
                target = data_val[1].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)

                for res,tar in zip(restored[0], target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

                if ii%num_val_samples==0:
                    break

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = (epoch-1)*data_epochs+(i//num_samples_per_epoch)
                torch.save({'epoch': best_epoch, 
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_best.pth"))
            current_epoch=(epoch-1)*data_epochs+(i//1000)
            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (current_epoch, psnr_val_rgb, best_epoch, best_psnr))

            writer.add_scalar("data/learning_rate", scheduler.get_last_lr()[0], current_epoch)
            writer.add_scalar("data/validation_psnr", psnr_val_rgb, current_epoch)
            scheduler.step()

            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
            print("------------------------------------------------------------------")

            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_latest.pth"))

            epoch_start_time = time.time()
            epoch_loss = 0
            train_id = 1
            model_restoration.train()

writer.close()
