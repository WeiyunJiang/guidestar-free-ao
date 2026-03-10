import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch.nn.functional as F

from poppy.zernike import zernike
from torch.utils.data import Dataset, random_split
from scipy.ndimage import shift
from PIL import Image
from torchvision.transforms import Compose, ToTensor

    

# Start timer (tic)
def tic():
    global start_time
    start_time = time.time()

# End timer and print elapsed time (toc)
def toc():
    if 'start_time' in globals():
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    # else:
    #     print("Timer not started. Call tic() first.")

def pad_to_size_tensor(tensor, final_size):
    """
    Pads a tensor symmetrically to a given final size.
    
    Parameters:
    tensor (torch.Tensor): Input tensor to be padded.
    final_size (tuple): The desired final size as (rows, cols).
    
    Returns:
    torch.Tensor: The padded tensor.
    """
    rows, cols = tensor.shape[-2], tensor.shape[-1]
    target_rows, target_cols = final_size
    
    if rows > target_rows or cols > target_cols:
        raise ValueError("Final size must be greater than or equal to the current size.")
    
    # Calculate the padding needed for rows and columns
    pad_top = (target_rows - rows) // 2
    pad_bottom = target_rows - rows - pad_top
    pad_left = (target_cols - cols) // 2
    pad_right = target_cols - cols - pad_left
    
    # Apply padding
    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return padded_tensor


class Places2(Dataset):
    def __init__(self, categories=['house'], split='train', resolution=(256, 256), 
                 downsampled=False, num_images=None, cache='none', 
                 psf_dir=None, psf_cache='none', phase_dir=None, phase_cache='none'):
        """
        Args:
            categories (list): List of categories of images (e.g., ['field-cultivated', 'house']).
            split (str): {train, val}.
            resolution (tuple): Desired image resolution.
            downsampled (bool): Whether to downsample the images.
            psf_dir (list of str): Paths to directories containing PSF files in .png format.
            psf_cache (str): How to cache the PSFs, {none, in_memory}.
            phase_dir (list of str): Paths to directories containing phase files in .npy format.
            phase_cache (str): How to cache the phases, {none, in_memory}.
        """
        super().__init__()
        print(f"Loading Places2 dataset with categories: {categories}, split: {split}")
        self.root = '/home/wj22/pupil/data/places2/places365_standard'
        self.fnames = []
        self.categories = categories
        self.resolution = resolution
        self.downsampled = downsampled
        self.num_images = num_images
        self.cache = cache
        self.psf_dirs = psf_dir if isinstance(psf_dir, list) else [psf_dir]
        self.psf_cache = psf_cache
        self.phase_dirs = phase_dir if isinstance(phase_dir, list) else [phase_dir]
        self.phase_cache = phase_cache

        # Load image paths from the appropriate .txt file for the specified split
        txt_file = os.path.join(self.root, f'{split}.txt')
        with open(txt_file) as f:
            content = f.readlines()

        # Filter by categories match
        for x in content:
            for category in self.categories:
                if x.startswith(f'{split}/{category}/'):
                    filename = x.strip()
                    file_path = os.path.join(self.root, filename)
                    if self.cache == 'none':
                        self.fnames.append(filename)
                    elif self.cache == 'in_memory':
                        self.fnames.append(
                            Image.open(file_path).convert('L')
                        )
                    else:
                        raise ValueError(f"Invalid cache option: {self.cache}")

        if self.num_images:
            self.fnames = self.fnames[:self.num_images]
        print(f"Number of images: {len(self.fnames)}")

        # Handle PSF and phase loading from multiple directories
        self.psf_paths = []
        self.phase_paths = []
        
        for psf_dir in self.psf_dirs:
            self.psf_paths.extend([os.path.join(psf_dir, psf) for psf in os.listdir(psf_dir) if psf.endswith('.png')])
        print(f"Number of PSF files: {len(self.psf_paths)}")
        
        for phase_dir in self.phase_dirs:
            self.phase_paths.extend([os.path.join(phase_dir, phase) for phase in os.listdir(phase_dir) if phase.endswith('.npy')])
        print(f"Number of phase files: {len(self.phase_paths)}")
        
        # Ensure PSF and phase paths are sorted and paired correctly
        self.psf_paths.sort()
        self.phase_paths.sort()

        if len(self.psf_paths) != len(self.phase_paths):
            raise ValueError("The number of PSF files must match the number of phase files.")

        if psf_cache == 'in_memory' and phase_cache == 'in_memory':
            self.psfs = []
            self.phases = []
            for psf_path, phase_path in zip(self.psf_paths, self.phase_paths):
                psf = Image.open(psf_path).convert('L')
                psf = np.array(psf) / 255.0  # Normalize to [0, 1]
                psf = torch.tensor(psf, dtype=torch.float32).unsqueeze(0)
                self.psfs.append(psf)

                phase = np.load(phase_path)
                phase = torch.tensor(phase, dtype=torch.float32).unsqueeze(0)
                self.phases.append(phase)
        elif psf_cache == 'none' and phase_cache == 'none':
            self.psfs = None
            self.phases = None
        else:
            raise ValueError("psf_cache and phase_cache options must match.")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # Load and transform the clean image
        if self.cache == 'none':
            filename = self.fnames[idx]
            file_path = os.path.join(self.root, filename)
            img = Image.open(file_path).convert('L')
        elif self.cache == 'in_memory':
            img = self.fnames[idx]

        if self.downsampled:
            img = img.resize(self.resolution)

        # Convert the image to a tensor
        transform = Compose([
            ToTensor(),
        ])
        img = transform(img)

        # Randomly select PSF and phase as pairs
        if self.psf_dirs and self.phase_dirs:
            random_idx = random.randint(0, len(self.psf_paths) - 1)
            if self.psf_cache == 'in_memory' and self.phase_cache == 'in_memory':
                psf = self.psfs[random_idx]
                phase = self.phases[random_idx]
            elif self.psf_cache == 'none' and self.phase_cache == 'none':
                psf_path = self.psf_paths[random_idx]
                psf = Image.open(psf_path).convert('L')
                psf = np.array(psf) / 255.0  # Normalize to [0, 1]
                psf = torch.tensor(psf, dtype=torch.float32).unsqueeze(0)

                phase_path = self.phase_paths[random_idx]
                phase = np.load(phase_path)
                phase = torch.tensor(phase, dtype=torch.float32).unsqueeze(0)
            else:
                raise ValueError("Invalid psf_cache or phase_cache option provided.")
        else:
            psf = None
            phase = None

        # Return the result dictionary
        result = {'clean_img': img}
        if psf is not None:
            result['psf'] = psf
        if phase is not None:
            result['phase'] = phase
        return result

def generate_uniform_random_vector(size, r1, r2, device='cpu'):
    """
    Generate a tensor of uniform random values between r1 and r2.

    Parameters:
        size (int): The length of the output vector.
        r1 (float): The lower bound of the uniform distribution.
        r2 (float): The upper bound of the uniform distribution.
        device (str): The device to place the tensor ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor of uniform random values between r1 and r2.
    """
    return r1 + (r2 - r1) * torch.rand(size, device=device)
    # return torch.empty(size, dtype=torch.float32, device=device).uniform_(r1, r2)

def generate_random_phase_mask_batched(size, max_order, num_phase_masks, sd, device='cpu'):
    """
    Generate a batch of random phase masks using Zernike polynomials with poppy and PyTorch.
    
    Parameters:
        size (int): The size of the phase mask (size x size).
        max_order (int): The maximum order of Zernike polynomials to include.
        num_phase_masks (int): The number of phase masks to generate in a batch.
        sd (torch.Tensor): A vector of standard deviations for each phase mask of length num_phase_masks.
        device (str): The device to which tensors should be moved ('cpu' or 'cuda').
        
    Returns:
        phase_masks (torch.Tensor): A batch of generated random phase masks of shape 
                                    (num_phase_masks, size, size).
    """
    # Ensure that sd is a tensor and has the correct length
    sd = torch.tensor(sd, device=device) if not isinstance(sd, torch.Tensor) else sd.to(device)
    assert sd.shape[0] == num_phase_masks, "The length of sd must be equal to num_phase_masks"
    
    # Create a coordinate grid and move to device
    y, x = torch.meshgrid(torch.linspace(-1, 1, size, device=device), 
                          torch.linspace(-1, 1, size, device=device), 
                          indexing='ij')
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    
    # Initialize the phase masks
    phase_masks = torch.zeros((num_phase_masks, size, size), device=device)
    
    # Generate random coefficients for Zernike polynomials for all masks in batch
    num_polynomials = (max_order + 1) * (max_order + 2) // 2
    
    # Use different standard deviations for each phase mask
    coefficients = torch.stack([torch.normal(mean=0.0, std=sd[i], size=(num_polynomials,), device=device)
                                for i in range(num_phase_masks)])
    
    # Mask rho and theta to only consider values inside the unit disk
    rho_masked = torch.where(rho <= 1, rho, torch.tensor(float('nan'), device=device))
    theta_masked = torch.where(rho <= 1, theta, torch.tensor(float('nan'), device=device))
    
    index = 0
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            if index < num_polynomials:
                # Use Zernike polynomial from poppy
                zernike_poly = zernike(n, m, rho=rho_masked.cpu().numpy(), theta=theta_masked.cpu().numpy())
                zernike_poly_torch = torch.tensor(zernike_poly, device=device)  # Convert back to PyTorch tensor
                
                # Add contribution from this Zernike polynomial to all phase masks in batch
                phase_masks += coefficients[:, index].view(num_phase_masks, 1, 1) * zernike_poly_torch
                index += 1
    
    # Mask out the values outside the unit disk

    phase_masks[:, rho > 1] = 0
    
    return phase_masks



def create_meshgrid(size, device='cpu'):
    """
    Create a meshgrid where the x and y values range from 0 to size using PyTorch.

    Parameters:
    size (int): The size of the meshgrid.
    device (str): The device to create the meshgrid on ('cpu' or 'cuda').

    Returns:
    tuple: Two 2D tensors representing the meshgrid coordinates.
    """
    x = torch.arange(0, size, device=device)
    y = torch.arange(0, size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return xx, yy

def is_point_in_triangle_vectorized(xx, yy, v1, v2, v3):
    """
    Vectorized check if points in the grid (xx, yy) are inside the triangle formed by v1, v2, v3.

    Parameters:
    xx, yy (torch.Tensor): The grid coordinates.
    v1, v2, v3 (tuple): Vertices of the triangle.

    Returns:
    torch.Tensor: A boolean tensor indicating whether each point is inside the triangle.
    """
    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = sign(xx, yy, v1[0], v1[1], v2[0], v2[1])
    d2 = sign(xx, yy, v2[0], v2[1], v3[0], v3[1])
    d3 = sign(xx, yy, v3[0], v3[1], v1[0], v1[1])

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return ~(has_neg & has_pos)

def generate_triangle_aperture(size, vertex1, vertex2, vertex3, device='cpu'):
    """
    Generate a binary aperture mask of a triangle given the vertices using PyTorch.

    Parameters:
    size (int): The size of the grid.
    vertex1, vertex2, vertex3 (tuple): Vertices of the triangle.
    device (str): The device to create the mask on ('cpu' or 'cuda').

    Returns:
    torch.Tensor: The binary aperture mask of the triangle.
    """
    xx, yy = create_meshgrid(size, device)
    
    # Perform a vectorized check to see which points are inside the triangle
    inside_triangle = is_point_in_triangle_vectorized(xx, yy, vertex1, vertex2, vertex3)
    
    # Create the aperture mask
    aperture = torch.zeros((size, size), device=device)
    aperture[inside_triangle] = 1.0
    
    return aperture

def phase2psf_batched(phase_batch, amp_batch, numPixels=256, device='cpu'):
    """
    Compute the Point Spread Function (PSF) from a batch of phase maps using the Fresnel diffraction formula.
    
    Parameters:
    phase_batch (torch.Tensor): The phase map tensor (batch_size, numPixels, numPixels).
    amp_batch (torch.Tensor): The pupil amplitude tensor (batch_size, numPixels, numPixels).
    numPixels (int): Number of pixels in the PSF.
    device (str): The device to run the computation on ('cpu' or 'cuda').
    
    Returns:
    torch.Tensor: The PSF tensor (batch_size, numPixels, numPixels).
    """
    pixelSize = 0.1  # microns
    Z0 = 376.73  # Ohms; impedance of free space
    power = 0.1  # Watts

    batch_size = phase_batch.shape[0]

    # Create the coordinate grid (same for all batches)
    x = torch.linspace(-pixelSize * numPixels / 2, pixelSize * numPixels / 2, steps=numPixels, device=device)
    dx = x[1] - x[0]  # Sampling period, microns
    fS = 1 / dx  # Spatial sampling frequency, inverse microns
    df = fS / numPixels  # Spacing between discrete frequency coordinates, inverse microns

    # Compute the normalization factor for the amplitude
    amp_squared = amp_batch ** 2
    norm_factor = torch.sum(amp_squared, dim=(-2, -1)) * (df ** 2) / Z0

    # Compute the wavefront: batch-wise computation
    pupil_batch = amp_batch * torch.exp(1j * phase_batch)

    # Normalize the pupil amplitude
    normed_pupil_batch = pupil_batch * torch.sqrt(power / norm_factor[:, None, None])

    # Perform batched FFT (2D FFT for each batch)
    pupil_batch_shifted = torch.fft.ifftshift(normed_pupil_batch, dim=(-2, -1))
    psf_batch = torch.fft.fftshift(torch.fft.fft2(pupil_batch_shifted), dim=(-2, -1)) * (df ** 2)

    # Compute the intensity (PSF)
    psf_intensity_batch = torch.abs(psf_batch) ** 2 / Z0

    # Normalize the PSF for each batch
    psf_intensity_batch /= torch.sum(psf_intensity_batch, dim=(-2, -1), keepdim=True)

    return psf_intensity_batch

def center_crop(img_tensor: torch.Tensor, crop_size: tuple) -> torch.Tensor:
    """
    Crop the center part of an image or batch of images to the specified size.

    This function handles several tensor shapes:
      - (H, W) for a single grayscale image.
      - (C, H, W) for a single color image.
      - (N, H, W) for a batch of grayscale images.
      - (N, C, H, W) for a batch of color images.

    Args:
        img_tensor (torch.Tensor): The image tensor to crop.
        crop_size (tuple): Desired crop size as (crop_height, crop_width).

    Returns:
        torch.Tensor: The center cropped tensor.
    """
    crop_H, crop_W = crop_size

    if img_tensor.dim() == 2:
        # Single grayscale image (H, W)
        H, W = img_tensor.shape
        start_H = (H - crop_H) // 2
        start_W = (W - crop_W) // 2
        return img_tensor[start_H:start_H+crop_H, start_W:start_W+crop_W]

    elif img_tensor.dim() == 3:
        # Could be (C, H, W) or (N, H, W)
        # Use a simple heuristic: if the first dimension is greater than 4, treat it as batch.
        if img_tensor.shape[0] > 4:
            # Assume (N, H, W) batch of grayscale images
            N, H, W = img_tensor.shape
            start_H = (H - crop_H) // 2
            start_W = (W - crop_W) // 2
            return img_tensor[:, start_H:start_H+crop_H, start_W:start_W+crop_W]
        else:
            # Assume (C, H, W) single image with channels
            C, H, W = img_tensor.shape
            start_H = (H - crop_H) // 2
            start_W = (W - crop_W) // 2
            return img_tensor[:, start_H:start_H+crop_H, start_W:start_W+crop_W]

    elif img_tensor.dim() == 4:
        # Batch of color images (N, C, H, W)
        N, C, H, W = img_tensor.shape
        start_H = (H - crop_H) // 2
        start_W = (W - crop_W) // 2
        return img_tensor[:, :, start_H:start_H+crop_H, start_W:start_W+crop_W]

    else:
        raise ValueError("Unsupported tensor shape. Expected a tensor with 2, 3, or 4 dimensions.")

def generate_amp_psf_and_phase(num_pairs, mask_type = 'triangle', numPixels=256, sd_curr=None, crop_psf_shape=64, device='cuda'):
    '''
    Generate amplitude, PSF, and phase for a batch of images.
    
    Parameters:
        num_pairs (int): Number of pairs to generate.
        mask_type (str): Type of mask to use ('triangle', 'circle').
        numPixels (int): Number of pixels in the PSF.
        sd_curr (torch.Tensor): Standard deviation for the phase mask.
        crop_psf_shape (int): Size to crop the PSF to.
        device (str): The device to run the computation on ('cpu' or 'cuda').
    Returns:
        torch.Tensor: The amplitude tensor (num_pairs, 1, numPixels, numPixels).
        torch.Tensor: The PSF tensor (num_pairs, 1, numPixels, numPixels).
        torch.Tensor: The phase tensor (num_pairs, 1, numPixels, numPixels).
    '''
    if 'triangle' in mask_type:
        # offsetx = 50
        # offsety = 30
        # vertex1 = (numPixels//2, offsetx)
        # vertex2 = (0 + offsety, (numPixels - 1.0) - offsetx)
        # vertex3 = ((numPixels - 1.0) - offsety, (numPixels - 1.0) - offsetx)
        # pupilMask = generate_triangle_aperture(numPixels, vertex1, vertex2, vertex3, device=device)
        
        
        offsetx = 0
        offsety = 0
        triangle_max_len = 144
        vertex1 = (triangle_max_len//2, offsetx)
        vertex2 = (0 + offsety, (triangle_max_len - 1.0) - offsetx)
        vertex3 = ((triangle_max_len - 1.0) - offsety, (triangle_max_len - 1.0) - offsetx)
        pupilMask = generate_triangle_aperture(triangle_max_len, vertex1, vertex2, vertex3, device=device)
        pupilMask = pad_to_size_tensor(pupilMask, (numPixels, numPixels))
    else:
        raise Exception(f'{mask_type} Not implemented')

    amps = torch.ones((num_pairs, numPixels, numPixels), device=device) * pupilMask
    # was using 256 for triangle # max_order=7
    entire_phase_masks = generate_random_phase_mask_batched(size=320, max_order=7, num_phase_masks=num_pairs, sd=sd_curr, device=device) 
    phase_masks = center_crop(entire_phase_masks, (numPixels, numPixels)) * pupilMask
    psfs = phase2psf_batched(phase_batch=phase_masks, amp_batch=amps, numPixels=numPixels, device=device)
    
    amps = amps.unsqueeze(1)
    psfs = psfs.unsqueeze(1)
    phase_masks = phase_masks.unsqueeze(1)
    if crop_psf_shape is not None:
        raise Exception('Not implemented')

        
    return amps, psfs, phase_masks

def generate_amp_psf_and_phase_kolmogorov(num_pairs, mask_type='triangle', numPixels=256,
                                           r0=None, crop_psf_shape=64, device='cuda'):
    '''
    Generate amplitude, PSF, and Kolmogorov-based phase for a batch of images.
    
    Parameters:
        num_pairs (int): Number of samples to generate.
        mask_type (str): Type of mask to use ('triangle').
        numPixels (int): Size of the phase/PSF.
        r0 (float): Fried parameter controlling turbulence strength.
        crop_psf_shape (int): Size to crop the PSF to (currently not implemented).
        device (str): 'cuda' or 'cpu'.

    Returns:
        amps: (num_pairs, 1, numPixels, numPixels)
        psfs: (num_pairs, 1, numPixels, numPixels)
        phase_masks: (num_pairs, 1, numPixels, numPixels)
    '''
    if 'triangle' in mask_type:
        offsetx = 0
        offsety = 0
        triangle_max_len = 144
        vertex1 = (triangle_max_len // 2, offsetx)
        vertex2 = (offsety, triangle_max_len - 1.0 - offsetx)
        vertex3 = (triangle_max_len - 1.0 - offsety, triangle_max_len - 1.0 - offsetx)
        pupilMask = generate_triangle_aperture(triangle_max_len, vertex1, vertex2, vertex3, device=device)
        pupilMask = pad_to_size_tensor(pupilMask, (numPixels, numPixels))
    else:
        raise Exception(f'{mask_type} Not implemented')

    # Generate amplitude mask for each sample
    amps = torch.ones((num_pairs, numPixels, numPixels), device=device) * pupilMask

    # Generate Kolmogorov phase
    full_phase_masks = generate_kolmogorov_phase_mask_batched(size=numPixels, r0_list=r0, device=device)
    phase_masks = full_phase_masks * pupilMask

    # Generate PSFs
    psfs = phase2psf_batched(phase_batch=phase_masks, amp_batch=amps,
                             numPixels=numPixels, device=device)

    # Add channel dimension
    amps = amps.unsqueeze(1)
    psfs = psfs.unsqueeze(1)
    phase_masks = phase_masks.unsqueeze(1)

    if crop_psf_shape is not None:
        raise Exception('PSF cropping not implemented')

    return amps, psfs, phase_masks
    
class PSF_Phase_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, mask_type='triangle', device='cuda'):
        print(f"Creating PSF_Phase_Wrapper with mask type: {mask_type}")
        self.transform = Compose([
            ToTensor(),
        ])
        self.dataset = dataset
        self.mask_type = mask_type
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['clean_img']
        img = self.transform(img)
        # C, H, W = img.size()
        # It is too heavy for cpu dataloading, I moved it to the training loop on gpu
        # sd_rand = generate_uniform_random_vector(size=1, r1=0.01, r2=1.0, device=self.device)
        # amps, psfs, phases = generate_amp_psf_and_phase(
        #                                         num_pairs=1, 
        #                                         mask_type=self.mask_type, 
        #                                         numPixels=H, 
        #                                         sd_curr=sd_rand,
        #                                         crop_psf_shape=None,
        #                                         device=self.device,
        #                                         )  
        
        # max_values = psfs.view(psfs.shape[0], -1).max(dim=1, keepdim=True)[0]  # Find max along each index
        # scaled_psfs = psfs / max_values.view(psfs.shape[0], 1, 1)
        # data_dict = {'clean_img': img, 'amps': amps, 'psfs': scaled_psfs, 'phases': phases}
        data_dict = {'clean_img': img}
        return data_dict
    

class PSFPhaseDataset(Dataset):
    def __init__(self, psf_dir, phase_dir, datasetSize, transform=None):
        """
        Custom Dataset for loading PSF and phase pairs with splitting into train/val/test sets.
        
        Parameters:
            psf_dir (str): Directory containing PSF `.npy` files.
            phase_dir (str): Directory containing phase `.npy` files.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (str): Which split to use ('train', 'val', 'test').
            split_ratios (tuple): Ratios for splitting the dataset into train, val, and test sets.
        """
        self.psf_dir = psf_dir
        self.phase_dir = phase_dir
        self.transform = transform
        self.datasetSize = datasetSize

        # List all files in the directories
        self.psf_files = sorted(os.listdir(psf_dir))
        self.phase_files = sorted(os.listdir(phase_dir))
        
        assert len(self.psf_files) == len(self.phase_files), "Mismatch between PSF and phase file counts."
        
        # Perform the split
        self.total_len = len(self.psf_files)

        self.files = list(zip(self.psf_files, self.phase_files))
        
        # train_files, val_files = random_split(
        #     list(zip(self.psf_files, self.phase_files)), 
        #     [train_len, val_len]
        # )
        
        # # Select the correct split
        # if split == "train":
        #     self.files = train_files
        # elif split == "val":
        #     self.files = val_files

        # else:
        #     raise ValueError("Split must be 'train', 'val'")
    
    def __len__(self):
        if self.total_len < self.datasetSize:
            print(f"Warning: dataset size is {self.total_len}, but requested size is {self.datasetSize}.")
            return self.total_len
        else:
            return self.datasetSize #len(self.files)
    
    def __getitem__(self, idx):
        # Get the PSF and phase filenames
        psf_file, phase_file = self.files[idx]
        
        # Load the PSF and phase from the respective files
        psf_path = os.path.join(self.psf_dir, psf_file)
        phase_path = os.path.join(self.phase_dir, phase_file)

        psf = np.load(psf_path)
        psf = psf / np.max(psf)  # Normalize the PSF
        
        phase = np.load(phase_path)
        
        # Convert to tensors
        psf = torch.tensor(psf, dtype=torch.float32)
        psf = psf.unsqueeze(0)  # Add channel dimension
        phase = torch.tensor(phase, dtype=torch.float32)
        phase = phase.unsqueeze(0)  # Add channel dimension
        # Apply transform if provided
        if self.transform:
            psf = self.transform(psf)
            phase = self.transform(phase)
        
        return psf, phase

class PSFPhaseDataset_Optical(Dataset):
    def __init__(self, psf_dir, phase_dir, datasetSize, transform=None):
        """
        Custom Dataset for loading PSF and phase pairs with splitting into train/val/test sets.

        Parameters:
            psf_dir (str): Directory containing PSF `.png` files.
            phase_dir (str): Directory containing phase `.npy` files.
            transform (callable, optional): Optional transform to be applied on a sample.
            datasetSize (int): Desired size of the dataset.
        """
        self.psf_dir = psf_dir
        self.phase_dir = phase_dir
        self.transform = transform
        self.datasetSize = datasetSize

        # List all files in the directories
        self.psf_files = sorted(os.listdir(psf_dir))
        self.phase_files = sorted(os.listdir(phase_dir))

        assert len(self.psf_files) == len(self.phase_files), "Mismatch between PSF and phase file counts."

        # Perform the split
        self.total_len = len(self.psf_files)

        self.files = list(zip(self.psf_files, self.phase_files))

    def __len__(self):
        if self.total_len < self.datasetSize:
            print(f"Warning: dataset size is {self.total_len}, but requested size is {self.datasetSize}.")
            return self.total_len
        else:
            return self.datasetSize

    def __getitem__(self, idx):
        # Get the PSF and phase filenames
        psf_file, phase_file = self.files[idx]

        # Load the PSF and phase from the respective files
        psf_path = os.path.join(self.psf_dir, psf_file)
        phase_path = os.path.join(self.phase_dir, phase_file)

        # Load the PSF as a grayscale image and normalize
        psf = Image.open(psf_path).convert('L')  # Convert to grayscale
        psf = np.array(psf, dtype=np.float32)
        psf = psf / 255.0  # Normalize to [0, 1]

        # Load the phase as a numpy array
        phase = np.load(phase_path)

        # Convert to tensors
        psf = torch.tensor(psf, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        phase = torch.tensor(phase, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Apply transform if provided
        if self.transform:
            psf = self.transform(psf)
            phase = self.transform(phase)

        return psf, phase

class PSFPhaseDataset_Optical_PNG(Dataset):
    def __init__(self, psf_dir, phase_dir, datasetSize, transform=None):
        """
        Custom Dataset for loading PSF and phase pairs with splitting into train/val/test sets.

        Parameters:
            psf_dir (str): Directory containing PSF `.png` files.
            phase_dir (str): Directory containing phase `.png` files.
            datasetSize (int): Desired size of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.psf_dir = psf_dir
        self.phase_dir = phase_dir
        self.transform = transform
        self.datasetSize = datasetSize

        # List all files in the directories
        self.psf_files = sorted(os.listdir(psf_dir))
        self.phase_files = sorted(os.listdir(phase_dir))

        assert len(self.psf_files) == len(self.phase_files), "Mismatch between PSF and phase file counts."

        # Pair the files
        self.files = list(zip(self.psf_files, self.phase_files))
        self.total_len = len(self.files)

    def __len__(self):
        if self.total_len < self.datasetSize:
            print(f"Warning: dataset size is {self.total_len}, but requested size is {self.datasetSize}.")
            return self.total_len
        else:
            return self.datasetSize

    def __getitem__(self, idx):
        # Get the PSF and phase filenames
        psf_file, phase_file = self.files[idx]

        # Construct full paths
        psf_path = os.path.join(self.psf_dir, psf_file)
        phase_path = os.path.join(self.phase_dir, phase_file)

        # Load the PSF as a grayscale image and normalize to [0, 1]
        psf = Image.open(psf_path).convert('L')
        psf = np.array(psf, dtype=np.float32)
        psf = psf / 255.0

        # Load the phase as a grayscale image, normalize and scale to [0, 2pi]
        phase_img = Image.open(phase_path).convert('L')
        phase = np.array(phase_img, dtype=np.float32)
        phase = (phase / 255.0) * (2 * np.pi)

        # Convert arrays to tensors and add a channel dimension
        psf = torch.tensor(psf, dtype=torch.float32).unsqueeze(0)
        phase = torch.tensor(phase, dtype=torch.float32).unsqueeze(0)

        # Apply transform if provided
        if self.transform:
            psf = self.transform(psf)
            phase = self.transform(phase)

        return psf, phase

class PSFPhaseBlurredDataset(Dataset):
    def __init__(self, psf_dir, phase_dir, blurred_images_dir, datasetSize, dataAugment=False, transform=None):
        """
        Custom Dataset for loading PSF, phase, and generating blurred images.
        
        Parameters:
            psf_dir (str): Directory containing PSF `.npy` files.
            phase_dir (str): Directory containing phase `.npy` files.
            blurred_images_dir (str): Directory containing blurred images.
            datasetSize (int): Number of samples in the dataset.
            dataAugment (bool): Whether to apply random shifts to the PSF.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.psf_dir = psf_dir
        self.phase_dir = phase_dir
        self.blurred_images_dir = blurred_images_dir
        self.transform = transform
        self.datasetSize = datasetSize
        self.dataAugment = dataAugment

        # List all files in the directories
        self.psf_files = sorted(os.listdir(psf_dir))
        self.phase_files = sorted(os.listdir(phase_dir))
        self.blurred_image_files = sorted(os.listdir(blurred_images_dir))
        
        assert len(self.psf_files) == len(self.phase_files) == len(self.blurred_image_files), \
            "Mismatch between PSF, phase, and image file counts."
        
        # Perform the split
        self.total_len = len(self.psf_files)

        self.files = list(zip(self.psf_files, self.phase_files, self.blurred_image_files))
        
    
    def __len__(self):
        if self.total_len < self.datasetSize:
            print(f"Warning: dataset size is {self.total_len}, but requested size is {self.datasetSize}.")
            return self.total_len
        else:
            return self.datasetSize
    
    def __getitem__(self, idx):
        # Get the PSF, phase, and image filenames
        psf_file, phase_file, blurred_image_file = self.files[idx]
        
        # Load the PSF and phase from the respective files
        psf_path = os.path.join(self.psf_dir, psf_file)
        phase_path = os.path.join(self.phase_dir, phase_file)
        blurred_image_path = os.path.join(self.blurred_images_dir, blurred_image_file)

        psf = np.load(psf_path)
        psf = psf / np.max(psf)  # Normalize the PSF
        phase = np.load(phase_path)
        blurred_image = np.array(Image.open(blurred_image_path).convert('L'))  # Convert image to grayscale (HxW)
        blurred_image = blurred_image / 255.0  # Normalize the image
        
        if self.dataAugment:
            # Random shift values for PSF augmentation
            shift_x = np.random.uniform(-5, 5)  # Random shift in x-direction
            shift_y = np.random.uniform(-5, 5)  # Random shift in y-direction

            # Shift the PSF
            psf = shift(psf, shift=[shift_y, shift_x], mode='nearest')


        # Convert to tensors
        psf = torch.tensor(psf, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        phase = torch.tensor(phase, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        blurred_image = torch.tensor(blurred_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Apply transform if provided
        if self.transform:
            blurred_image = self.transform(blurred_image)
            psf = self.transform(psf)
            phase = self.transform(phase)
        
        return psf, phase, blurred_image


if __name__ == "__main__":
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # torch.set_num_threads(4)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print(torch.cuda.get_device_name(0))

    # Example usage
    place2_dataset = Places2(category='house', split='train', resolution=(256, 256), downsampled=False, num_images=5000, cache='in_memory')
    print(len(place2_dataset))
    psf_phase_dataset_train = PSF_Phase_Wrapper(dataset=place2_dataset, mask_type='triangle', device='cpu')
    dataloader_train = torch.utils.data.DataLoader(psf_phase_dataset_train, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)
    for epoch in range(100):
        for i, data in enumerate(dataloader_train):
            print(epoch, i)
            clean_img = data['clean_img'].to(device)
            
            batch_size, C, H, W = clean_img.size()
            sd_rand = generate_uniform_random_vector(size=batch_size, r1=0.01, r2=1.0, device=device)
            amps, psfs, phases = generate_amp_psf_and_phase(
                                                    num_pairs=batch_size, 
                                                    mask_type='triangle', 
                                                    numPixels=H, 
                                                    sd_curr=sd_rand,
                                                    crop_psf_shape=None,
                                                    device=device,
                                                    )  
            
            max_values = psfs.view(psfs.shape[0], -1).max(dim=1, keepdim=True)[0]  # Find max along each index
            scaled_psfs = psfs / max_values.view(psfs.shape[0], 1, 1, 1)
    
            # psfs = data['psfs'].to(device)
            # print(psfs.shape)
            



    
    