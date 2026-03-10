# Generating psf for training psf Unet
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from dataio import generate_uniform_random_vector, generate_amp_psf_and_phase

def set_seed(seed):
    """Set the random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_data(amp, psf, phase, i, split, split_dirs):
    split_dir = split_dirs[split]
    
    amp_path = os.path.join(split_dir, 'amp', f'{i:08}_amp.png')
    psf_path = os.path.join(split_dir, 'psf', f'{i:08}_psf.png')
    phase_path = os.path.join(split_dir, 'phase', f'{i:08}_phase.npy')

    amp_np = amp.squeeze().detach().cpu().numpy()
    psf_np = psf.squeeze().detach().cpu().numpy()
    phase_np = phase.squeeze().detach().cpu().numpy()

    Image.fromarray((amp_np * 255).astype(np.uint8)).save(amp_path)
    Image.fromarray((np.clip(psf_np, 0, 1) * 255).astype(np.uint8)).save(psf_path)
    np.save(phase_path, phase_np)

def process_split(split_name, total_num, batch_size, mask_type, r1, r2, device, split_dirs):
    num_batches = total_num // batch_size
    if num_batches == 0 and total_num > 0:
        num_batches = 1
        batch_size = total_num

    for i in tqdm(range(num_batches), desc=f"Processing {split_name}"):
        sd_rand = generate_uniform_random_vector(size=batch_size, r1=r1, r2=r2, device=device)
        amps, psfs, phases = generate_amp_psf_and_phase(
                                    num_pairs=batch_size, 
                                    mask_type=mask_type, 
                                    numPixels=256, 
                                    sd_curr=sd_rand,
                                    crop_psf_shape=None,
                                    device=device,
                                    ) 
        
        # Normalize the PSF batch-wise
        max_values = psfs.view(psfs.shape[0], -1).max(dim=1, keepdim=True)[0]
        psfs = psfs / max_values.view(psfs.shape[0], 1, 1, 1)
        
        for batch_idx in range(batch_size):
            save_data(
                  amps[batch_idx], 
                  psfs[batch_idx], 
                  phases[batch_idx], 
                  i * batch_size + batch_idx,
                  split_name,
                  split_dirs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PSF datasets for Phase Unet")
    parser.add_argument('--gpu', type=str, default='2', help='GPU ID to use')
    parser.add_argument('--mask_type', type=str, default='triangle', help='Aperture mask type')
    parser.add_argument('--num_psf', type=int, default=200000, help='Total number of training PSFs')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for generation')
    parser.add_argument('--r1', type=float, default=0.01, help='Lower bound for random vector')
    parser.add_argument('--r2', type=float, default=1.5, help='Upper bound for random vector')
    parser.add_argument('--seed', type=int, default=390, help='Random seed for reproducibility')
    parser.add_argument('--output_root', type=str, default='./data/', help='Root directory for data')
    parser.add_argument('--name', type=str, default=None, help='Custom folder name (overrides default naming)')

    args = parser.parse_args()

    # Set visibility before any torch calls
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} (Visible ID: {args.gpu})')
    
    set_seed(args.seed)

    # Use custom name if provided, else generate a descriptive one
    if args.name:
        folder_name = args.name
    else:
        folder_name = f"{args.mask_type}_{args.num_psf//1000}k_r{args.r1}-{args.r2}_s{args.seed}"

    output_dir = os.path.join(args.output_root, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    split_dirs = {
        'train': os.path.join(output_dir, 'train'),
        'val': os.path.join(output_dir, 'val'),
        'test': os.path.join(output_dir, 'test')
    }

    # Initialize subdirectories
    for split_path in split_dirs.values():
        for sub in ['psf', 'phase', 'amp']:
            os.makedirs(os.path.join(split_path, sub), exist_ok=True)

    # Start generation
    process_split('train', args.num_psf, args.batch_size, args.mask_type, args.r1, args.r2, device, split_dirs)
    process_split('val', 3000, min(3000, args.batch_size), args.mask_type, args.r1, args.r2, device, split_dirs)
    process_split('test', 3000, min(3000, args.batch_size), args.mask_type, args.r1, args.r2, device, split_dirs)

    print(f"\n✅ Dataset generation complete.")
    print(f"📍 Saved to: {output_dir}")