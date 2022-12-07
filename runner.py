import os
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as TF

import utils
from layers import *
from DeepRFT_MIMO import DeepRFT as mynet
from get_parameter_number import get_parameter_number
from skimage.metrics import peak_signal_noise_ratio as psnr_loss

# %%

doc_dir = "./publaynet/"
result_dir = "./results/"
weights = "./DeepRFT-MIMO-v1/DeepRFT/model_GoPro.pth"

num_res = 8         # num of resblocks, [Small, Med, PLus]=[4, 8, 20]
win = 256           # window size, [GoPro, HIDE, RealBlur]=256, [DPDD]=512
gpus = 0            # CUDA_VISIBLE_DEVICES

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

# %%

# model_restoration = mynet()
model_restoration = mynet(num_res=num_res, inference=True)
# print number of model
get_parameter_number(model_restoration)
# utils.load_checkpoint(model_restoration, weights)
utils.load_checkpoint_compress_doconv(model_restoration, weights)
print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# %%

from customDataSet import CustomDataset

data = CustomDataset(doc_dir,(400,400),False)
dataloader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
original_dir = os.path.join(result_dir,"original")
blurred_dir = os.path.join(result_dir,"blurred")
restored_dir = os.path.join(result_dir,"restored")

utils.mkdir(result_dir)
utils.mkdir(original_dir)
utils.mkdir(blurred_dir)
utils.mkdir(restored_dir)

# %%

psnr_val_rgb = []
psnr = 0

with torch.no_grad():
    psnr_list = []
    ssim_list = []
    topil = TF.ToPILImage()
    for ii, data_test in enumerate(tqdm(dataloader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        
        input_      = data_test[0].cuda()
        gt          = data_test[1].cpu().detach()
        filenames   = data_test[2]
        
        _, _, Hx, Wx = input_.shape
        input_re, batch_list = window_partitionx(input_, win)
        restored = model_restoration(input_re)
        restored = window_reversex(restored, win, Hx, Wx, batch_list)

        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = img_as_ubyte(restored[batch])

            utils.save_img((os.path.join(restored_dir, filenames[batch]+'.png')), topil(restored_img))
            utils.save_img((os.path.join(original_dir, filenames[batch]+'.png')), topil(gt[batch]))
            utils.save_img((os.path.join(blurred_dir, filenames[batch]+'.png')), topil(input_[batch]))
            try:
                psnr = psnr_loss(restored_img, gt[batch])
                print("sklean psnr", psnr)
                psnr_val_rgb.append(psnr)
            except:
                print("sklearn psnr failed")
                pass
            try:
                psnr = float(utils.torchPSNR(restored_img, gt[batch]).cpu().detach().numpy())
                print("torch psnr", psnr)
                psnr_val_rgb.append(psnr)
            except:
                print("torch psnr failed")
                pass

psnr = sum(psnr_val_rgb) / len(psnr_val_rgb)
print("PSNR: %f" % psnr)

with open("PSNRs.pkl","wb") as f:
    pickle.dump(psnr_val_rgb,f)
