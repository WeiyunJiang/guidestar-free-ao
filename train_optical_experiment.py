import argparse
import os
import torch
import torch.nn as nn
import time
import numpy as np
import random
import torch.nn.functional as F
import os

# and then import other packages. 
from network import Unet
from dataio import PSFPhaseDataset_Optical
from torch.utils.data import DataLoader
from utils import cond_mkdir, log_phase_psf_and_upsample_to_tensorboard, resume_training, torch_ifft2, torch_ifft2_angle, torch_convolve2d, create_disk_tensor, upsample_psf, add_gaussian_noise
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

# Limit the number of threads PyTorch uses for parallel computations
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# torch.set_num_threads(4)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='6')
parser.add_argument('--psf_dir_train', type=str, default='./data/triangle_200k_psf_0.01_1.5_new/train/psf')
parser.add_argument('--phase_dir_train', type=str, default='./data/triangle_200k_psf_0.01_1.5_new/train/phase')
parser.add_argument('--psf_dir_valid', type=str, default='./data/triangle_200k_psf_0.01_1.5_new/val/psf')
parser.add_argument('--phase_dir_valid', type=str, default='./data/triangle_200k_psf_0.01_1.5_new/val/phase')


parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=2e-5)  
parser.add_argument('--loss_type', type=str, default='grad')
parser.add_argument('--use_ifft', type=bool, default=False)
parser.add_argument('--name', type=str, default='unet_200k_norm_triangle_lr_2e-5_0.01_0.05_new') 
parser.add_argument('--train_size', type=int, default=200000) #200000
parser.add_argument('--valid_size', type=int, default=3000)
parser.add_argument('--radius', type=float, default=1)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--use_upsample', type=bool, default=False)
parser.add_argument('--gaussian_noise', type=bool, default=False)
args = parser.parse_args()


resume = False
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.cuda.get_device_name(0))

seed = 88

random.seed(seed)          # Python's random module
np.random.seed(seed)       # NumPy
torch.manual_seed(seed)    # PyTorch

# If using CUDA:
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

model = Unet(in_channels=1,out_channels=1,nf0=32,num_down=5,max_channels=512,use_dropout=False,outermost_linear=True).to(device)
model.train()
# Create the dataset for training
train_dataset = PSFPhaseDataset_Optical(args.psf_dir_train, args.phase_dir_train, datasetSize=args.train_size)
train_dataloader = DataLoader(train_dataset, shuffle=True,batch_size=args.batch_size, pin_memory=True, num_workers=8, persistent_workers=True)
# Create the dataset for validation
val_dataset = PSFPhaseDataset_Optical(args.psf_dir_valid, args.phase_dir_valid, datasetSize=args.valid_size)
val_dataloader = DataLoader(val_dataset, shuffle=False,batch_size=args.batch_size, pin_memory=True, num_workers=8, persistent_workers=True)

optim = torch.optim.Adam(lr=args.lr, params=model.parameters(), amsgrad=False)


# Initialize amp as None
amp = None

if 'triangle' in args.name:
    amp_path = './data/triangle_200k_psf_0.01_1.5_new/val/amp/00000000_amp.png'
    if os.path.exists(amp_path):
        # Read image and convert to grayscale ('L' mode)
        amp_pil = Image.open(amp_path).convert('L')
        
        # Convert to numpy and normalize to [0, 1]
        amp_np = np.array(amp_pil).astype(np.float32) / 255.0
        
        # Convert to tensor and move to device
        # Adding [None, None, ...] ensures it has (1, 1, H, W) shape if needed
        amp = torch.from_numpy(amp_np).to(device)
        print(f"Successfully loaded triangle mask via PIL: {amp_path}")
    else:
        print(f"Warning: Mask not found at {amp_path}")

if args.loss_type == "grad":
    criterion = nn.MSELoss(reduction='mean')
elif args.loss_type == 'l2':
    criterion = nn.MSELoss(reduction='mean')
elif args.loss_type == 'l1':
    criterion = nn.L1Loss(reduction='mean')
else:
    raise ValueError("Loss type must be 'grad'")
model_dir = os.path.join('models', args.name)
cond_mkdir(model_dir)
summaries_dir = os.path.join(model_dir, 'summaries')
cond_mkdir(summaries_dir)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')
cond_mkdir(checkpoints_dir)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (f'\n\nTraining model with {num_parameters} parameters\n\n')
checkpoint_path = os.path.join(checkpoints_dir, 'model_last.pth')
if resume:
    start_epoch = resume_training(model, optim, checkpoint_path, device=device)
else:
    start_epoch = 0
writer = SummaryWriter(summaries_dir)
total_steps = 0


train_losses = []
val_best = 1e10
for epoch in range(start_epoch, args.epochs):
    for step, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{args.epochs}"):
        psf = batch[0].to(device)
        if args.use_ifft:
            psf = torch_ifft2_angle(psf)
    
        phase = batch[1].to(device)

        # create disk
        # disk_tensor = create_disk_tensor(args.radius, psf.shape[0], (16, 16))
        # disk_tensor = disk_tensor.to(device)

        # psf_conv = torch_convolve2d(psf, disk_tensor, device)
        if args.use_upsample:
            psf_upsampled = upsample_psf(psf, args.scale_factor, target_shape=psf.shape[2:4]) 
        else:
            psf_upsampled = psf
        if args.gaussian_noise:
            psf_noisy = add_gaussian_noise(psf_upsampled, std_range=(0.0, 0.08))
        else:
            psf_noisy = psf_upsampled
        start_time = time.time()
        pred_phase = model(psf_noisy)
        if args.loss_type == "grad":
            x_kernel = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).to(device).type(torch.float32)
            y_kernel = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).to(device).type(torch.float32)
            pred_grad_x = F.conv2d(pred_phase, x_kernel, padding=1)
            pred_grad_y = F.conv2d(pred_phase, y_kernel, padding=1)
            target_grad_x = F.conv2d(phase, x_kernel, padding=1)
            target_grad_y = F.conv2d(phase, y_kernel, padding=1)
            assert pred_grad_x.shape == target_grad_x.shape, f"Mismatch in shape: pred {pred_grad_x.shape}, target {target_grad_x.shape}"
            assert pred_grad_y.shape == target_grad_y.shape, f"Mismatch in shape: pred {pred_grad_y.shape}, target {target_grad_y.shape}"
           
            loss = criterion(pred_grad_x, target_grad_x) + criterion(pred_grad_y, target_grad_y)
        elif args.loss_type == 'l2':
            loss = criterion(pred_phase, phase)
        elif args.loss_type == 'l1':
            loss = criterion(pred_phase, phase)
        else:
            raise ValueError("Loss type must be 'grad'")
        writer.add_scalar("train_loss",loss,total_steps)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        

        
        if not total_steps % 3900:
            torch.cuda.empty_cache()
            # tqdm.write("Epoch %d, outer loss %0.6f, iteration time %0.6f" % (epoch, loss, time.time() - start_time))
            log_phase_psf_and_upsample_to_tensorboard(writer, pred_phase, phase, psf, psf_noisy, total_steps, tag="train", amp=amp)
            

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':optim.state_dict(),
                        'loss':loss,
                        'total_steps':total_steps},
                        os.path.join(checkpoints_dir,f"model_{total_steps}.pth"))
            
            model.eval()
            with torch.no_grad():
                val_loss = []
                for step, batch in enumerate(val_dataloader):
                    psf = batch[0].to(device)
                    if args.use_ifft:
                        psf = torch_ifft2_angle(psf)
                    phase = batch[1].to(device)
                    # create disk
                    # disk_tensor = create_disk_tensor(args.radius, psf.shape[0], (16, 16))
                    # disk_tensor = disk_tensor.to(device)

                    # psf_conv = torch_convolve2d(psf, disk_tensor, device)
                    if args.use_upsample:
                        psf_upsampled = upsample_psf(psf, args.scale_factor, target_shape=psf.shape[2:4]) 
                    else:
                        psf_upsampled = psf
                    if args.gaussian_noise:
                        psf_noisy = add_gaussian_noise(psf_upsampled, std_range=(0.0, 0.08))
                    else:
                        psf_noisy = psf_upsampled
                    pred_phase = model(psf_noisy)
                    if args.loss_type == "grad":
                        x_kernel = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).to(device).type(torch.float32)
                        y_kernel = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).to(device).type(torch.float32)
                        pred_grad_x = F.conv2d(pred_phase, x_kernel, padding=1)
                        pred_grad_y = F.conv2d(pred_phase, y_kernel, padding=1)
                        target_grad_x = F.conv2d(phase, x_kernel, padding=1)
                        target_grad_y = F.conv2d(phase, y_kernel, padding=1)
                        loss = criterion(pred_grad_x, target_grad_x) + criterion(pred_grad_y, target_grad_y)
                        val_loss.append(loss.detach().cpu().numpy())
                    elif args.loss_type == 'l2':
                        loss = criterion(pred_phase, phase)
                        val_loss.append(loss.detach().cpu().numpy())
                    elif args.loss_type == 'l1':
                        loss = criterion(pred_phase, phase)
                        val_loss.append(loss.detach().cpu().numpy())
                    else:
                        raise ValueError("Loss type must be 'grad'")
                    
                    log_phase_psf_and_upsample_to_tensorboard(writer, pred_phase, phase, psf, psf_noisy, total_steps, tag="val", amp=amp)
                writer.add_scalar("val_loss",np.mean(val_loss),total_steps)
                if val_best > np.mean(val_loss):
                    val_best = np.mean(val_loss)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':optim.state_dict(),
                                'loss':loss,
                                'total_steps':total_steps},
                                os.path.join(checkpoints_dir,"model_best.pth"))
                print("eval:{}".format(np.mean(val_loss)))
            if args.loss_type == 'grad':
                del psf, phase, pred_phase, x_kernel, y_kernel, pred_grad_x, pred_grad_y, target_grad_x, target_grad_y, loss
            elif args.loss_type == 'l2':
                del psf, phase, pred_phase, loss
            elif args.loss_type == 'l1':
                del psf, phase, pred_phase, loss
            torch.cuda.empty_cache()
            model.train()

        total_steps = total_steps+1

model.eval()
with torch.no_grad():
    val_loss = []
    for step, batch in enumerate(val_dataloader):
        psf = batch[0].to(device)
        if args.use_ifft:
            psf = torch_ifft2_angle(psf)
        phase = batch[1].to(device)
        # create disk
        # disk_tensor = create_disk_tensor(args.radius, psf.shape[0], (16, 16))
        # disk_tensor = disk_tensor.to(device)

        # psf_conv = torch_convolve2d(psf, disk_tensor, device)
        if args.use_upsample:
            psf_upsampled = upsample_psf(psf, args.scale_factor, target_shape=psf.shape[2:4]) 
        else:
            psf_upsampled = psf
        if args.gaussian_noise:
            psf_noisy = add_gaussian_noise(psf_upsampled, std_range=(0.0, 0.08))
        else:
            psf_noisy =psf_upsampled
        pred_phase = model(psf_noisy)
        if args.loss_type == "grad":
            x_kernel = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).to(device).type(torch.float32)
            y_kernel = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).to(device).type(torch.float32)
            pred_grad_x = F.conv2d(pred_phase, x_kernel, padding=1)
            pred_grad_y = F.conv2d(pred_phase, y_kernel, padding=1)
            target_grad_x = F.conv2d(phase, x_kernel, padding=1)
            target_grad_y = F.conv2d(phase, y_kernel, padding=1)
            loss = criterion(pred_grad_x, target_grad_x) + criterion(pred_grad_y, target_grad_y)
            val_loss.append(loss.detach().cpu().numpy())
        elif args.loss_type == 'l2':
            loss = criterion(pred_phase, phase)
        elif args.loss_type == 'l1':
            loss = criterion(pred_phase, phase)
        else:
            raise ValueError("Loss type must be 'grad'")
        log_phase_psf_and_upsample_to_tensorboard(writer, pred_phase, phase, psf, psf_noisy, total_steps, tag="val", amp=amp)
    writer.add_scalar("val_loss",np.mean(val_loss),total_steps)
    if val_best > np.mean(val_loss):
        val_best = np.mean(val_loss)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optim.state_dict(),
                    'loss':loss,
                    'total_steps':total_steps},
                    os.path.join(checkpoints_dir,"model_best.pth"))
    print("eval:{}".format(np.mean(val_loss)))
torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'loss':loss,
            'total_steps':total_steps},
            os.path.join(checkpoints_dir,"model_last.pth"))
