import argparse
import os
import torch
import torch.nn as nn
import time
import numpy as np
import random
import torch.nn.functional as F

from network import Unet, BlurKernelUNet64x64
from dataio import PSFPhaseBlurredDataset, Places2, PSF_Phase_Wrapper, generate_uniform_random_vector, generate_amp_psf_and_phase
from torch.utils.data import DataLoader
from utils import cond_mkdir, log_phase_psf_and_blurred_to_tensorboard, resume_training, create_central_mask, masked_mse_loss, torch_convolve2d,  central_crop, patchify_blurred_img, add_gaussian_noise
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# torch.set_num_threads(4)
mask_type = 'triangle'
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--train_size', type=int, default=100000)
parser.add_argument('--valid_size', type=int, default=2000)


parser.add_argument('--psf_dir_train', nargs='+', type=str, 
                    default=['/home/wj22/pupil/data/triangle_1m_psf_1.5_finetune_new/train/psf', '/home/wj22/pupil/data/triangle_1m_psf_1.3_finetune_new/train/psf', '/home/wj22/pupil/data/triangle_1m_psf_1.1_finetune_new/train/psf', '/home/wj22/pupil/data/triangle_1m_psf_0.9_finetune_new/train/psf',
                    '/home/wj22/pupil/data/triangle_2m_psf_0.7_finetune_new/train/psf', '/home/wj22/pupil/data/1m_psf_0.5_finetune_new/train/psf', '/home/wj22/pupil/data/500k_psf_0.3_finetune_new/train/psf',
                    '/home/wj22/pupil/data/triangle_200k_psf_0.1_finetune_new/train/psf','/home/wj22/pupil/data/triangle_100k_psf_0.01_finetune_new/train/psf'],
                    help="Paths to training PSF directories.")
parser.add_argument('--phase_dir_train', nargs='+', type=str, 
                    default=['/home/wj22/pupil/data/triangle_1m_psf_1.5_finetune_new/train/phase', '/home/wj22/pupil/data/triangle_1m_psf_1.3_finetune_new/train/phase', '/home/wj22/pupil/data/triangle_1m_psf_1.1_finetune_new/train/phase', '/home/wj22/pupil/data/triangle_1m_psf_0.9_finetune_new/train/phase',
                    '/home/wj22/pupil/data/triangle_2m_psf_0.7_finetune_new/train/phase', '/home/wj22/pupil/data/1m_psf_0.5_finetune_new/train/phase', '/home/wj22/pupil/data/500k_psf_0.3_finetune_new/train/phase',
                    '/home/wj22/pupil/data/triangle_200k_psf_0.1_finetune_new/train/phase', '/home/wj22/pupil/data/triangle_100k_psf_0.01_finetune_new/train/phase'], 
                    help="Paths to training phase directories.")

parser.add_argument('--psf_dir_val', nargs='+', type=str, 
                    default=['/home/wj22/pupil/data/triangle_2m_psf_0.7_finetune_new/val/psf'],
                    help="Paths to validation PSF directories.")
parser.add_argument('--phase_dir_val', nargs='+', type=str, 
                    default=['/home/wj22/pupil/data/triangle_2m_psf_0.7_finetune_new/val/phase'],
                    help="Paths to validation phase directories.")




parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--epochs', type=int, default=500000)
parser.add_argument('--lr', type=float, default=2e-5)  
parser.add_argument('--loss_type', type=str, default='grad')
# parser.add_argument('--mask_type', type=str, default='triangle')
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--patches_shuffle', type=bool, default=False)
parser.add_argument('--gaussian_noise', type=bool, default=False)
parser.add_argument('--normalized_psf', type=bool, default=False)
parser.add_argument('--name', type=str, default=f'only_phase_unet_lr_2e-5_places2_100k_img_7.8m_psf_finetune_100k_img_7.8m_psf_1.5_grad_loss_max_channels_4096_nf0_128_finetune_{mask_type}_bz_200_no_normalized_psf_2000_val')
parser.add_argument('--checkpoint_path_phase', type=str, default='/home/wj22/pupil/models/unet_200k_norm_triangle_lr_2e-5_0722_0.01_1.5/checkpoints/model_85800.pth')
parser.add_argument('--checkpoint_path_psf', type=str, default='/home/wj22/pupil/models_ker_est/patched_ker_est_unet_lr_2e-4_places2_100k_img_7.8m_psf_1.5_num_down_5_max_channels_4096_nf0_128_bz_256_triangle/checkpoints/model_684000.pth')
args = parser.parse_args()


resume = False
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.cuda.get_device_name(0))

seed = 17

random.seed(seed)          # Python's random module
np.random.seed(seed)       # NumPy
torch.manual_seed(seed)    # PyTorch

# If using CUDA:
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

model_phase = Unet(in_channels=1,out_channels=1,nf0=32,num_down=5,max_channels=512,use_dropout=False,outermost_linear=True).to(device)

checkpoint_path_phase = args.checkpoint_path_phase
checkpoint_phase = torch.load(checkpoint_path_phase, map_location=device)
# Load the model state dict
model_phase.load_state_dict(checkpoint_phase['model_state_dict'])
model_phase.to(device)

model_psf = Unet(in_channels=16,out_channels=1,nf0=128,num_down=5,max_channels=4096,use_dropout=False,outermost_linear=True).to(device)

checkpoint_path_psf = args.checkpoint_path_psf
checkpoint_psf = torch.load(checkpoint_path_psf, map_location=device)
# Load the model state dict
model_psf.load_state_dict(checkpoint_psf['model_state_dict'])
model_psf.to(device)


categories = [
    "courthouse",
    "hotel-outdoor",
    "embassy",
    "castle",
    "palace",
    "tower",
    "amphitheater",
    "bridge",
    "campus",
    "driveway",
    "farm",
    "highway",
    "plaza",
    "roof_garden",
    "schoolhouse",
    "train_station-platform",
    "pavilion",
    "temple-asia",
    "synagogue-outdoor",
    "restaurant_patio"
]

# Create the dataset for training
place2_dataset_train = Places2(categories=categories, 
                               split='train', 
                               resolution=(256, 256), 
                               downsampled=False, 
                               num_images=args.train_size, 
                               cache='none',
                               psf_dir=args.psf_dir_train, 
                               psf_cache='none',
                               phase_dir=args.phase_dir_train,
                               phase_cache='none')
dataloader_train = torch.utils.data.DataLoader(place2_dataset_train, 
                                               batch_size=args.batch_size, 
                                               shuffle=True, 
                                               pin_memory=True, 
                                               num_workers=8, 
                                               persistent_workers=True)
# Create the dataset for validation
### Use train for special toy example
### Need to CHANGE val to train later
place2_dataset_val = Places2(categories=categories, 
                             split='val', 
                             resolution=(256, 256), 
                             downsampled=False, 
                             num_images=args.valid_size, 
                             cache='none',
                             psf_dir=args.psf_dir_val, 
                             psf_cache='none',
                             phase_dir=args.phase_dir_val,
                             phase_cache='none')

dataloader_val = torch.utils.data.DataLoader(place2_dataset_val, 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             pin_memory=True, 
                                             num_workers=8,
                                             persistent_workers=True)

optim = torch.optim.Adam(lr=args.lr, params=model_phase.parameters(), amsgrad=False)

if args.loss_type == "grad":
    criterion = nn.MSELoss(reduction='mean')
elif args.loss_type == "l1":
    criterion = nn.L1Loss(reduction='mean')
elif args.loss_type == "l2":
    criterion = nn.MSELoss(reduction='mean')
else:
    raise ValueError("Loss type must be 'grad'")
model_dir = os.path.join('models_joint', args.name)
cond_mkdir(model_dir)
summaries_dir = os.path.join(model_dir, 'summaries')
cond_mkdir(summaries_dir)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')
cond_mkdir(checkpoints_dir)

num_parameters = sum(p.numel() for p in model_phase.parameters() if p.requires_grad) + sum(p.numel() for p in model_psf.parameters() if p.requires_grad)
print (f'\n\nTraining model with {num_parameters} parameters\n\n')
checkpoint_path = os.path.join(checkpoints_dir, 'model_last.pth')
if resume:
    raise ValueError("Resume not implemented")
    # start_epoch = resume_training(model, optim, checkpoint_path, device=device)
else:
    start_epoch = 0
writer = SummaryWriter(summaries_dir)
total_steps = 0


train_losses = []
val_best = 1e10
for epoch in range(start_epoch, args.epochs):
    for step, batch in tqdm(enumerate(dataloader_train), desc=f"Epoch {epoch + 1}/{args.epochs}"):
        clean_img = batch['clean_img'].to(device)
        psfs = batch['psf'].to(device)
        phases = batch['phase'].to(device)
        B, C, H, W = psfs.size()
        # crop psf to 64x64
        psfs = central_crop(psfs, crop_size=args.patch_size, original_size=H)
        
        max_values =  psfs.contiguous().view(psfs.shape[0], -1).max(dim=1, keepdim=True)[0]  # Find max along each index
        psfs = psfs / max_values.view(psfs.shape[0], 1, 1, 1) # Normalize the PSF from 0 to 1
        blurred_img = torch_convolve2d(clean_img, psfs, device)
        if args.gaussian_noise:
            blurred_img = add_gaussian_noise(blurred_img, std_range=(0.0, 0.01))
        patches = patchify_blurred_img(blurred_img, patch_size=args.patch_size, shuffle=args.patches_shuffle)
        # print(f'max blurred_image: {torch.max(blurred_img)}')
        # print(f'min blurred_image: {torch.min(blurred_img)}')
        # print(f'min psf: {torch.min(psfs)}')
        # print(f'max psf: {torch.max(psfs)}')
        # print(psfs.shape)
        # print(blurred_img.shape)
        
        pred_psf = model_psf(patches)
        if not args.normalized_psf:
            pred_psf = torch.clamp(pred_psf, 0.0, 1.0) # some values of the output from model_psf are negative and over 1
        else:
            B = pred_psf.shape[0]
            pred_psf_flat = pred_psf.view(B, -1)  # (B, 4096)

            min_vals = pred_psf_flat.min(dim=1, keepdim=True)[0]  # (B, 1)
            max_vals = pred_psf_flat.max(dim=1, keepdim=True)[0]  # (B, 1)

            pred_psf_flat = pred_psf_flat - min_vals  # broadcast per row

            # Error check: any max_val <= 0 in the batch
            if (max_vals <= 0).any():
                idx = (max_vals <= 0).nonzero(as_tuple=True)[0]
                raise ValueError(f"max_val <= 0 after normalization for batch index/indices: {idx.tolist()}, max_val: {max_vals[idx].tolist()}")

            pred_psf_flat = pred_psf_flat / max_vals  # broadcast per row
            pred_psf_flat = torch.clamp(pred_psf_flat, 0.0, 1.0)

            pred_psf = pred_psf_flat.view(B, 1, 64, 64)

        padding = (96, 96, 96, 96)  # (left, right, top, bottom)
        padded_psf = F.pad(pred_psf, padding, mode='constant', value=0)
        pred_phase = model_phase(padded_psf)

        if args.loss_type == "grad":
            x_kernel = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).to(device).type(torch.float32)
            y_kernel = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).to(device).type(torch.float32)
            pred_grad_x = F.conv2d(pred_phase, x_kernel, padding=1)
            pred_grad_y = F.conv2d(pred_phase, y_kernel, padding=1)
            target_grad_x = F.conv2d(phases, x_kernel, padding=1)
            target_grad_y = F.conv2d(phases, y_kernel, padding=1)
            loss = criterion(pred_grad_x, target_grad_x) + criterion(pred_grad_y, target_grad_y)
        elif args.loss_type == "l1":
            loss = criterion(pred_phase, phases)
        elif args.loss_type == "l2":
            # loss = criterion(pred_psf, psf) + criterion(pred_phase, phase)
            loss = criterion(pred_phase, phases)
        else:
            raise ValueError("Loss type must be 'grad'")
        writer.add_scalar("train_loss",loss,total_steps)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        

        
        if not total_steps % 3000:# 2350:
            torch.cuda.empty_cache()
            # tqdm.write("Epoch %d, outer loss %0.6f, iteration time %0.6f" % (epoch, loss, time.time() - start_time))
            log_phase_psf_and_blurred_to_tensorboard(writer, pred_phase, phases, pred_psf, psfs, blurred_img, total_steps, tag="train")
            

            torch.save({'epoch': epoch,
                        'model_state_dict': model_phase.state_dict(),
                        'optimizer_state_dict':optim.state_dict(),
                        'loss':loss,
                        'total_steps':total_steps},
                        os.path.join(checkpoints_dir,f"model_phase_{total_steps}.pth"))
            torch.save({'epoch': epoch,
                        'model_state_dict': model_psf.state_dict(),
                        'optimizer_state_dict':optim.state_dict(),
                        'loss':loss,
                        'total_steps':total_steps},
                        os.path.join(checkpoints_dir,f"model_psf_{total_steps}.pth"))
            
            model_phase.eval()
            model_psf.eval()
            with torch.no_grad():
                val_loss = []
                for step, batch in enumerate(dataloader_val):
                    clean_img = batch['clean_img'].to(device)
                    psfs = batch['psf'].to(device)
                    phases = batch['phase'].to(device)
                    B, C, H, W = psfs.size()
                    # crop psf to 64x64
                    psfs = central_crop(psfs, crop_size=args.patch_size, original_size=H)
                    
                    max_values = psfs.contiguous().view(psfs.shape[0], -1).max(dim=1, keepdim=True)[0]  # Find max along each index
                    psfs = psfs / max_values.view(psfs.shape[0], 1, 1, 1) # Normalize the PSF from 0 to 1
                    blurred_img = torch_convolve2d(clean_img, psfs, device)
                    patches = patchify_blurred_img(blurred_img, patch_size=args.patch_size, shuffle=args.patches_shuffle)

                    pred_psf = model_psf(patches)
                    if not args.normalized_psf:
                        pred_psf = torch.clamp(pred_psf, 0.0, 1.0) # some values of the output from model_psf are negative and over 1
                    else:
                        B = pred_psf.shape[0]
                        pred_psf_flat = pred_psf.view(B, -1)  # (B, 4096)

                        min_vals = pred_psf_flat.min(dim=1, keepdim=True)[0]  # (B, 1)
                        max_vals = pred_psf_flat.max(dim=1, keepdim=True)[0]  # (B, 1)

                        pred_psf_flat = pred_psf_flat - min_vals  # broadcast per row

                        # Error check: any max_val <= 0 in the batch
                        if (max_vals <= 0).any():
                            idx = (max_vals <= 0).nonzero(as_tuple=True)[0]
                            raise ValueError(f"max_val <= 0 after normalization for batch index/indices: {idx.tolist()}, max_val: {max_vals[idx].tolist()}")

                        pred_psf_flat = pred_psf_flat / max_vals  # broadcast per row
                        pred_psf_flat = torch.clamp(pred_psf_flat, 0.0, 1.0)

                        pred_psf = pred_psf_flat.view(B, 1, 64, 64)

                    padding = (96, 96, 96, 96)  # (left, right, top, bottom)
                    padded_psf = F.pad(pred_psf, padding, mode='constant', value=0)
                    pred_phase = model_phase(padded_psf)

                    if args.loss_type == "grad":
                        x_kernel = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).to(device).type(torch.float32)
                        y_kernel = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).to(device).type(torch.float32)
                        pred_grad_x = F.conv2d(pred_phase, x_kernel, padding=1)
                        pred_grad_y = F.conv2d(pred_phase, y_kernel, padding=1)
                        target_grad_x = F.conv2d(phases, x_kernel, padding=1)
                        target_grad_y = F.conv2d(phases, y_kernel, padding=1)
                        loss = criterion(pred_grad_x, target_grad_x) + criterion(pred_grad_y, target_grad_y)
                    elif args.loss_type == "l1":
                        loss = criterion(pred_phase, phases)
                    elif args.loss_type == "l2":
                        # loss = criterion(pred_psf, psf) + criterion(pred_phase, phase)
                        loss =  criterion(pred_phase, phases)
                    else:
                        raise ValueError("Loss type must be 'grad'")
                    val_loss.append(loss.detach().cpu().numpy())
                    log_phase_psf_and_blurred_to_tensorboard(writer, pred_phase, phases, pred_psf, psfs, blurred_img, total_steps, tag="val")

                writer.add_scalar("val_loss",np.mean(val_loss),total_steps)
                if val_best > np.mean(val_loss):
                    val_best = np.mean(val_loss)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model_phase.state_dict(),
                                'optimizer_state_dict':optim.state_dict(),
                                'loss':loss,
                                'total_steps':total_steps},
                                os.path.join(checkpoints_dir,"model_phase_best.pth"))
                    torch.save({'epoch': epoch,
                                'model_state_dict': model_psf.state_dict(),
                                'optimizer_state_dict':optim.state_dict(),
                                'loss':loss,
                                'total_steps':total_steps},
                                os.path.join(checkpoints_dir,"model_psf_best.pth"))
                print("eval:{}".format(np.mean(val_loss)))
            if args.loss_type == "grad":
                del x_kernel, y_kernel, pred_grad_x, pred_grad_y, target_grad_x, target_grad_y
            del blurred_img, psfs, pred_psf, loss
            torch.cuda.empty_cache()
            model_phase.train()
            model_psf.train()

        total_steps = total_steps+1

model_phase.eval()
model_psf.eval()
with torch.no_grad():
    val_loss = []
    for step, batch in enumerate(dataloader_val):
        clean_img = batch['clean_img'].to(device)
        psfs = batch['psf'].to(device)
        phases = batch['phase'].to(device)
        B, C, H, W = psfs.size()
        # crop psf to 64x64
        psfs = central_crop(psfs, crop_size=args.patch_size, original_size=H)
        
        
        max_values = psfs.contiguous().view(psfs.shape[0], -1).max(dim=1, keepdim=True)[0]  # Find max along each index
        psfs = psfs / max_values.view(psfs.shape[0], 1, 1, 1) # Normalize the PSF from 0 to 1
        blurred_img = torch_convolve2d(clean_img, psfs, device)
        patches = patchify_blurred_img(blurred_img, patch_size=args.patch_size, shuffle=args.patches_shuffle)

        pred_psf = model_psf(patches)
        if not args.normalized_psf:
            pred_psf = torch.clamp(pred_psf, 0.0, 1.0) # some values of the output from model_psf are negative and over 1
        else:
            B = pred_psf.shape[0]
            pred_psf_flat = pred_psf.view(B, -1)  # (B, 4096)

            min_vals = pred_psf_flat.min(dim=1, keepdim=True)[0]  # (B, 1)
            max_vals = pred_psf_flat.max(dim=1, keepdim=True)[0]  # (B, 1)

            pred_psf_flat = pred_psf_flat - min_vals  # broadcast per row

            # Error check: any max_val <= 0 in the batch
            if (max_vals <= 0).any():
                idx = (max_vals <= 0).nonzero(as_tuple=True)[0]
                raise ValueError(f"max_val <= 0 after normalization for batch index/indices: {idx.tolist()}, max_val: {max_vals[idx].tolist()}")

            pred_psf_flat = pred_psf_flat / max_vals  # broadcast per row
            pred_psf_flat = torch.clamp(pred_psf_flat, 0.0, 1.0)

            pred_psf = pred_psf_flat.view(B, 1, 64, 64)        
        padding = (96, 96, 96, 96)  # (left, right, top, bottom)
        padded_psf = F.pad(pred_psf, padding, mode='constant', value=0)
        pred_phase = model_phase(padded_psf)

        if args.loss_type == "grad":
            x_kernel = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).to(device).type(torch.float32)
            y_kernel = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).to(device).type(torch.float32)
            pred_grad_x = F.conv2d(pred_phase, x_kernel, padding=1)
            pred_grad_y = F.conv2d(pred_phase, y_kernel, padding=1)
            target_grad_x = F.conv2d(phases, x_kernel, padding=1)
            target_grad_y = F.conv2d(phases, y_kernel, padding=1)
            loss = criterion(pred_grad_x, target_grad_x) + criterion(pred_grad_y, target_grad_y)
        elif args.loss_type == "l1":
            loss = criterion(pred_phase, phases)
        elif args.loss_type == "l2":
            # loss = criterion(pred_psf, psf) + criterion(pred_phase, phase)
            loss =  criterion(pred_phase, phases)
        else:
            raise ValueError("Loss type must be 'grad'")
        val_loss.append(loss.detach().cpu().numpy())
        log_phase_psf_and_blurred_to_tensorboard(writer, pred_phase, phases, pred_psf, psfs, blurred_img, total_steps, tag="val")
    writer.add_scalar("val_loss",np.mean(val_loss),total_steps)
    if val_best > np.mean(val_loss):
        val_best = np.mean(val_loss)
        torch.save({'epoch': epoch,
                    'model_state_dict': model_phase.state_dict(),
                    'optimizer_state_dict':optim.state_dict(),
                    'loss':loss,
                    'total_steps':total_steps},
                    os.path.join(checkpoints_dir,"model_phase_best.pth"))
        torch.save({'epoch': epoch,
                    'model_state_dict': model_psf.state_dict(),
                    'optimizer_state_dict':optim.state_dict(),
                    'loss':loss,
                    'total_steps':total_steps},
                    os.path.join(checkpoints_dir,"model_psf_best.pth"))
    print("eval:{}".format(np.mean(val_loss)))
torch.save({'epoch': epoch,
            'model_state_dict': model_phase.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'loss':loss,
            'total_steps':total_steps},
            os.path.join(checkpoints_dir,"model_phase_last.pth"))
torch.save({'epoch': epoch,
            'model_state_dict': model_psf.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'loss':loss,
            'total_steps':total_steps},
            os.path.join(checkpoints_dir,"model_psf_last.pth"))          
