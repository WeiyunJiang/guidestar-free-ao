import os
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn.functional as F

from scipy.integrate import simps
from numpy.fft import fft2, fftshift, ifftshift
from poppy.zernike import zernike


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log_phase_and_psf_to_tensorboard(writer, pred_phase, gt_phase, psf, total_steps, tag="train"):
    """
    Log the predicted phase, ground truth phase, and PSF to TensorBoard using the HSV colormap.
    
    Parameters:
        writer (SummaryWriter): The TensorBoard SummaryWriter object.
        pred_phase (Tensor): The predicted phase tensor (2D tensor).
        gt_phase (Tensor): The ground truth phase tensor (2D tensor).
        psf (Tensor): The Point Spread Function tensor (2D tensor).
        epoch (int): The current epoch number.
        tag (str): The tag to group the images under in TensorBoard.
    """
    # Convert tensors to numpy arrays for plotting
    pred_phase_np = pred_phase[0].squeeze().detach().cpu().numpy()
    gt_phase_np = gt_phase[0].squeeze().detach().cpu().numpy()
    psf_np = psf[0].squeeze().detach().cpu().numpy()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot predicted phase with HSV colormap and color bar
    im_pred = axes[0].imshow(pred_phase_np, cmap='hsv')
    axes[0].set_title('Predicted Phase')
    axes[0].axis('off')
    fig.colorbar(im_pred, ax=axes[0])
    
    # Plot ground truth phase with HSV colormap and color bar
    im_gt = axes[1].imshow(gt_phase_np, cmap='hsv')
    axes[1].set_title('Ground Truth Phase')
    axes[1].axis('off')
    fig.colorbar(im_gt, ax=axes[1])
    
    # Plot PSF with a suitable colormap (e.g., 'viridis') and color bar
    im_psf = axes[2].imshow(psf_np, cmap='viridis')
    axes[2].set_title('Point Spread Function (PSF)')
    axes[2].axis('off')
    fig.colorbar(im_psf, ax=axes[2])
    
    # Save the figure to a buffer
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Log the image to TensorBoard
    writer.add_image(f'{tag}/Pred_GT_PSF', img, total_steps, dataformats='HWC')
    
    # Close the figure to free up memory
    plt.close(fig)

def log_phase_psf_and_upsample_to_tensorboard(writer, pred_phase, gt_phase, psf, psf_upsample, total_steps, tag="train", amp=None):
    # Convert tensors to numpy
    pred_phase_np = pred_phase[0].squeeze().detach().cpu().numpy()
    gt_phase_np = gt_phase[0].squeeze().detach().cpu().numpy()
    psf_np = psf[0].squeeze().detach().cpu().numpy()
    psf_upsample_np = psf_upsample[0].squeeze().detach().cpu().numpy()
    
    # Apply mask if available
    if amp is not None:
        # Ensure amp is a numpy array and matches the phase shape
        if torch.is_tensor(amp):
            amp = amp.squeeze().detach().cpu().numpy()
        
        # Normalize amp to [0, 1] if it's a 0-255 image
        if amp.max() > 1:
            amp = amp / 255.0
            
        pred_phase_np = pred_phase_np * amp
        gt_phase_np = gt_phase_np * amp
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Plot predicted phase with HSV colormap and color bar
    im_pred = axes[0].imshow(pred_phase_np, cmap='hsv')
    axes[0].set_title('Predicted Phase')
    axes[0].axis('off')
    fig.colorbar(im_pred, ax=axes[0])
    
    # Plot ground truth phase with HSV colormap and color bar
    im_gt = axes[1].imshow(gt_phase_np, cmap='hsv')
    axes[1].set_title('Ground Truth Phase')
    axes[1].axis('off')
    fig.colorbar(im_gt, ax=axes[1])
    
    # Plot original PSF with a suitable colormap (e.g., 'viridis') and color bar
    im_psf = axes[2].imshow(psf_np, cmap='viridis')
    axes[2].set_title('Original PSF')
    axes[2].axis('off')
    fig.colorbar(im_psf, ax=axes[2])
    
    # Plot upsampled PSF with a suitable colormap (e.g., 'viridis') and color bar
    im_psf_upsample = axes[3].imshow(psf_upsample_np, cmap='viridis')
    axes[3].set_title('Upsampled PSF')
    axes[3].axis('off')
    fig.colorbar(im_psf_upsample, ax=axes[3])
    
    # Save the figure to a buffer
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Log the image to TensorBoard
    writer.add_image(f'{tag}/Phase_PSF_and_Upsample', img, total_steps, dataformats='HWC')
    
    # Close the figure to free up memory
    plt.close(fig)

def log_psf_and_blurred_to_tensorboard(writer, pred_psf, gt_psf, blurred_image, total_steps, tag="train"):
    """
    Log the predicted phase, ground truth phase, and PSF to TensorBoard using the HSV colormap.
    
    Parameters:
        writer (SummaryWriter): The TensorBoard SummaryWriter object.
        pred_phase (Tensor): The predicted phase tensor (2D tensor).
        gt_phase (Tensor): The ground truth phase tensor (2D tensor).
        psf (Tensor): The Point Spread Function tensor (2D tensor).
        epoch (int): The current epoch number.
        tag (str): The tag to group the images under in TensorBoard.
    """
    # Convert tensors to numpy arrays for plotting
    pred_psf_np = pred_psf[0].squeeze().detach().cpu().numpy()
    gt_psf_np = gt_psf[0].squeeze().detach().cpu().numpy()
    blurred_image_np = blurred_image[0].squeeze().detach().cpu().numpy()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot predicted phase with HSV colormap and color bar
    im_pred = axes[0].imshow(pred_psf_np, cmap='viridis')
    axes[0].set_title('Predicted PSF')
    axes[0].axis('off')
    fig.colorbar(im_pred, ax=axes[0])
    
    # Plot ground truth phase with HSV colormap and color bar
    im_gt = axes[1].imshow(gt_psf_np, cmap='viridis')
    axes[1].set_title('Ground Truth PSF')
    axes[1].axis('off')
    fig.colorbar(im_gt, ax=axes[1])
    
    # Plot PSF with a suitable colormap (e.g., 'viridis') and color bar
    im_psf = axes[2].imshow(blurred_image_np, cmap='gray')
    axes[2].set_title('Blurred Image')
    axes[2].axis('off')
    fig.colorbar(im_psf, ax=axes[2])
    
    # Save the figure to a buffer
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Log the image to TensorBoard
    writer.add_image(f'{tag}/Pred_GT_PSF', img, total_steps, dataformats='HWC')
    
    # Close the figure to free up memory
    plt.close(fig)

def log_phase_psf_and_blurred_to_tensorboard(writer, pred_phase, gt_phase, pred_psf, gt_psf, blurred_image, total_steps, tag="train"):
    """
    Log the predicted phase, ground truth phase, and PSF to TensorBoard using the HSV colormap.
    
    Parameters:
        writer (SummaryWriter): The TensorBoard SummaryWriter object.
        pred_phase (Tensor): The predicted phase tensor (2D tensor).
        gt_phase (Tensor): The ground truth phase tensor (2D tensor).
        psf (Tensor): The Point Spread Function tensor (2D tensor).
        epoch (int): The current epoch number.
        tag (str): The tag to group the images under in TensorBoard.
    """
    # Convert tensors to numpy arrays for plotting
    pred_psf_np = pred_psf[0].squeeze().detach().cpu().numpy()
    gt_psf_np = gt_psf[0].squeeze().detach().cpu().numpy()

    pred_phase_np = pred_phase[0].squeeze().detach().cpu().numpy()
    gt_phase_np = gt_phase[0].squeeze().detach().cpu().numpy()

    blurred_image_np = blurred_image[0].squeeze().detach().cpu().numpy()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    
    # Plot predicted phase with HSV colormap and color bar
    im_pred = axes[0].imshow(pred_psf_np, cmap='viridis')
    axes[0].set_title('Predicted PSF')
    axes[0].axis('off')
    fig.colorbar(im_pred, ax=axes[0])
    
    # Plot ground truth phase with HSV colormap and color bar
    im_gt = axes[1].imshow(gt_psf_np, cmap='viridis')
    axes[1].set_title('Ground Truth PSF')
    axes[1].axis('off')
    fig.colorbar(im_gt, ax=axes[1])

    im_pred_phase = axes[2].imshow(pred_phase_np, cmap='hsv')
    axes[2].set_title('Predicted Phase')
    axes[2].axis('off')
    fig.colorbar(im_pred_phase, ax=axes[2])
    
    # Plot ground truth phase with HSV colormap and color bar
    im_gt_phase = axes[3].imshow(gt_phase_np, cmap='hsv')
    axes[3].set_title('Ground Truth Phase')
    axes[3].axis('off')
    fig.colorbar(im_gt_phase, ax=axes[3])

    
    # Plot PSF with a suitable colormap (e.g., 'viridis') and color bar
    im_blur = axes[4].imshow(blurred_image_np, cmap='gray')
    axes[4].set_title('Blurred Image')
    axes[4].axis('off')
    fig.colorbar(im_blur, ax=axes[4])
    
    # Save the figure to a buffer
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Log the image to TensorBoard
    writer.add_image(f'{tag}/Pred_GT_PSF_Phase', img, total_steps, dataformats='HWC')
    
    # Close the figure to free up memory
    plt.close(fig)


def resume_training(model, optimizer, checkpoint_path, device):
    """
    Resumes training from a checkpoint.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        int: The epoch to resume training from.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Load the optimizer state dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    # Load the last epoch
    start_epoch = checkpoint['epoch']
    
    # Load any additional data (if any)
    if 'loss' in checkpoint:
        loss = checkpoint['loss']
        print(f"Resuming from epoch {start_epoch} with loss {loss}")
    else:
        print(f"Resuming from epoch {start_epoch}")

    return start_epoch

def is_point_in_triangle(pt, v1, v2, v3):
    """
    Check if a point is inside a triangle using barycentric coordinates.

    Parameters:
    pt (tuple): The point to check, given as (x, y).
    v1 (tuple): The first vertex of the triangle, given as (x, y).
    v2 (tuple): The second vertex of the triangle, given as (x, y).
    v3 (tuple): The third vertex of the triangle, given as (x, y).

    Returns:
    bool: True if the point is inside the triangle, False otherwise.
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def create_meshgrid(size):
    """
    Create a meshgrid where the x and y values range from 0 to size.

    Parameters:
    size (int): The size of the meshgrid.

    Returns:
    tuple: Two 2D arrays representing the meshgrid coordinates.
    """
    x = np.arange(0, size)
    y = np.arange(0, size)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def generate_triangle_aperture(size, vertex1, vertex2, vertex3):
    aperture = np.zeros((size, size))
    xx, yy = create_meshgrid(size)
    # Check if each point in the grid is inside the triangle
    for j in range(size):
        for i in range(size):
            point = (yy[j, i], xx[j, i])
            if is_point_in_triangle(point, vertex1, vertex2, vertex3):
                aperture[j, i] = 1.0
    return aperture

def pad_to_size(arr, final_size):
    """
    Pads an array symmetrically to a given final size.
    
    Parameters:
    arr (numpy.ndarray): Input array to be padded.
    final_size (tuple): The desired final size as (rows, cols).
    
    Returns:
    numpy.ndarray: The padded array.
    """
    rows, cols = arr.shape
    target_rows, target_cols = final_size
    
    if rows > target_rows or cols > target_cols:
        raise ValueError("Final size must be greater than or equal to the current size.")
    
    # Calculate the padding needed for rows and columns
    pad_top = (target_rows - rows) // 2
    pad_bottom = target_rows - rows - pad_top
    pad_left = (target_cols - cols) // 2
    pad_right = target_cols - cols - pad_left
    
    # Apply padding
    padded_array = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded_array

# phase2psf
def phase2psf(phase, amp):
    """
    ********
    To-Do: add normalization 
    ********
    Compute the Point Spread Function (PSF) from the phase map using the Fresnel diffraction formula.

    Parameters:
    phase (np.array): The phase map tensor (2D tensor).
    amp (np.array): The pupil amplitude tensor (2D tensor).

    Returns:
    np.array: The Point Spread Function (PSF) tensor (2D tensor).
    """
    pixelSize  = 0.1    # microns
    numPixels  = 256   # Number of pixels in the camera; keep this even

    Z0         = 376.73 # Ohms; impedance of free space
    power      = 0.1    # Watts
    x = np.linspace(-pixelSize * numPixels / 2, pixelSize * numPixels / 2, num = numPixels, endpoint = True)
    dx = x[1] - x[0]    # Sampling period, microns
    fS = 1 / dx         # Spatial sampling frequency, inverse microns
    df = fS / numPixels # Spacing between discrete frequency coordinates, inverse microns

    norm_factor = simps(simps(np.abs(amp)**2, dx = df), dx = df) / Z0
    # Compute the wavefront
    pupil = amp * np.exp(1j * phase)

    normedPupil = pupil * np.sqrt(power / norm_factor)
    psf_a = fftshift(fft2(ifftshift(normedPupil))) * df**2

    image = np.abs(psf_a)**2 / Z0
    image /= np.sum(image) # Normalize the PSF
    return image

def pad_kernel_to_size(kernel, desired_size=256):
    """
    Pads a given kernel with zeros to make it a specified size.

    Parameters:
    kernel (np.ndarray): The input kernel (2D array) to be padded.
    desired_size (int): The size of the output array (default is 256).

    Returns:
    np.ndarray: The padded kernel with the desired size.
    """
    kernel_size = kernel.shape[0]
    
    # Calculate the padding needed
    padding = ((desired_size - kernel_size) // 2, (desired_size - kernel_size) // 2)
    
    # Pad the kernel with zeros
    padded_kernel = np.pad(kernel, (padding, padding), mode='constant', constant_values=0)
    
    # Adjust for odd difference in size if necessary
    if padded_kernel.shape[0] != desired_size or padded_kernel.shape[1] != desired_size:
        padded_kernel = np.pad(padded_kernel, ((desired_size - padded_kernel.shape[0], 0 ), 
                                               (0, desired_size - padded_kernel.shape[1])), mode='constant', constant_values=0)
    
    return padded_kernel

def generate_random_phase_mask(size, max_order, sd=1.0):
    """
    Generate a random phase mask using Zernike polynomials with poppy.
    
    Parameters:
        size (int): The size of the phase mask (size x size).
        max_order (int): The maximum order of Zernike polynomials to include.
        
    Returns:
        phase_mask (ndarray): The generated random phase mask.
    """
    # Create a coordinate grid
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Initialize the phase mask
    phase_mask = np.zeros_like(rho)
    
    # Generate random coefficients for Zernike polynomials
    num_polynomials = (max_order + 1) * (max_order + 2) // 2
    coefficients = np.random.normal(loc=0, scale=sd, size=num_polynomials) #np.random.randn(num_polynomials)
    rho_masked = np.where(rho <= 1, rho, np.nan)
    theta_masked = np.where(rho <= 1, theta, np.nan)
    
    index = 0
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            if index < len(coefficients):
                # print(f'n {n}')
                # print(f'm {m}')
                # print(f'rho {rho}')
                # print(f'theta {theta}')
                zernike_poly = zernike(n, m, rho=rho_masked, theta=theta_masked)
                phase_mask += coefficients[index] * zernike_poly
                index += 1
    
    # Mask out the values outside the unit disk
    phase_mask[rho > 1] = 0
    
    return phase_mask



def create_central_mask(image_size=256, mask_size=50):
    """
    Create a mask for the central square of size mask_size x mask_size in an image of size image_size x image_size.
    
    Args:
        image_size (int): The size of the image (assumed to be square).
        mask_size (int): The size of the central mask (assumed to be square).
        
    Returns:
        mask (numpy array): A binary mask with the central square filled with 1's and the rest 0's.
    """
    # Initialize the mask with all zeros
    mask = np.zeros((image_size, image_size))

    # Calculate the coordinates for the central square
    start = (image_size - mask_size) // 2
    end = start + mask_size

    # Set the central region to 1
    mask[start:end, start:end] = 1

    return mask

def masked_mse_loss(pred, target, mask):
    """
    Compute the Mean Squared Error (MSE) loss only on the masked region.
    
    Args:
        pred (torch.Tensor): The predicted image or output (batch_size, channels, height, width).
        target (torch.Tensor): The ground truth image or output (batch_size, channels, height, width).
        mask (torch.Tensor): The binary mask (1, 1, height, width) with 1's in the area to compute the loss and 0's elsewhere.
        
    Returns:
        torch.Tensor: The masked MSE loss.
    """
    # Ensure mask has the same shape as the pred and target (broadcast if necessary)
    mask = mask.expand_as(pred)
    
    # Compute the squared difference
    squared_diff = (pred - target) ** 2
    
    # Apply the mask to the squared difference
    masked_squared_diff = squared_diff * mask
    
    # Compute the mean only over the masked region
    masked_loss = masked_squared_diff.sum() / mask.sum()  # Mask sum gives number of valid pixels
    
    return masked_loss


def create_disk_tensor(radius, num_disks, shape):
    """
    Create a 3D tensor where each slice along the first dimension contains a disk 
    of the specified radius and array shape.

    Args:
    radius (int): The radius of the disk.
    num_disks (int): Number of disks to create.
    shape (tuple): Shape of each 2D disk (height, width).

    Returns:
    torch.Tensor: A 3D tensor where each slice corresponds to a disk.
    """
    def create_disk(radius, shape):
        """
        Create a disk (circular mask) of the specified radius centered in an array of the specified shape.

        Args:
        radius (int): The radius of the disk.
        shape (tuple): The shape of the array (height, width).

        Returns:
        np.array: An array with the same specified shape, where points inside the radius are 1, and others are 0.
        """
        height, width = shape
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        distance_from_center = (x - center_x)**2 + (y - center_y)**2

        # Create the disk mask
        mask = distance_from_center <= radius**2

        return mask.astype(np.float32)  # Convert boolean mask to a float32 array

    # Create a single disk
    single_disk = create_disk(radius, shape)

    # Replicate the single disk along the first dimension to create the 3D tensor
    tensor = np.repeat(single_disk[np.newaxis, ...], num_disks, axis=0)

    # Convert to PyTorch tensor
    tensor = torch.tensor(tensor, dtype=torch.float32)

    # Add a channel dimension to match (batch size, 1, H, W)
    tensor = tensor.unsqueeze(1)

    return tensor

def upsample_psf(psf, scale_factor, target_shape):
    """
    Upsample the PSF tensor by a specified scale factor and crop the central portion to the target shape.

    Args:
    psf (torch.Tensor): Input tensor of shape (batch size, 1, H, W).
    scale_factor (float): Scale factor for upsampling.
    target_shape (tuple): Target shape (H, W) for the cropped tensor.

    Returns:
    torch.Tensor: Cropped tensor of shape (batch size, 1, target_H, target_W).
    """
    # Upsample the PSF tensor
    upsampled = F.interpolate(psf, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    # Calculate cropping indices
    _, _, up_H, up_W = upsampled.shape
    target_H, target_W = target_shape
    start_H = (up_H - target_H) // 2
    start_W = (up_W - target_W) // 2

    # Crop the central portion
    cropped = upsampled[:, :, start_H:start_H + target_H, start_W:start_W + target_W]

    return cropped


def torch_convolve2d(image, psf, device):
    '''
    Perform 2D convolution of an image with a Point Spread Function (PSF) using PyTorch.
    
    Args:
        image (torch.Tensor): The input image tensor (batch_size, channels, height, width).
        psf (torch.Tensor): The Point Spread Function (PSF) tensor (batch_size, channels, height, width).
        device (str): The device to move the tensors to ('cuda' or 'cpu').
    '''
    # Ensure the image and PSF are PyTorch tensors
    image_tensor = image.float().to(device)  # Convert image to tensor and move to the specified device
    psf_tensor = psf.float().to(device)  # Convert PSF to tensor and add channel dimension

    psf_sums = psf_tensor.sum(dim=[2, 3], keepdim=True) 
    psf_tensor = psf_tensor / psf_sums  # Normalize the PSF
    # Calculate the padding needed for reflective padding
    pad_size = (psf_tensor.shape[-1] - 1)// 2  # Assuming square PSF
    
    # Apply reflective padding to the image
    if psf_tensor.shape[-1] % 2 == 0: # psf is even size
        padding = (pad_size, pad_size+1, pad_size, pad_size+1)
    else:
        padding = (pad_size, pad_size, pad_size, pad_size)
    image_tensor_padded = F.pad(image_tensor, padding, mode='reflect').transpose(0, 1) # (1, 16, 256, 256)
    
    # Perform the 2D convolution
    result = F.conv2d(image_tensor_padded, psf_tensor, padding=0, groups=image_tensor.shape[0])  # No padding, since we manually padded the image
    
    # Remove the extra dimensions and return as NumPy array
    return result.transpose(0, 1)

def torch_ifft2(psf_tensor):
    '''
    Perform the inverse Fourier Transform on a Point Spread Function (PSF) tensor using PyTorch.
    psf_tensor (torch.Tensor): The Point Spread Function (PSF) tensor (batch_size, channels, height, width).
    '''
    psf_complex = torch.complex(psf_tensor, torch.zeros_like(psf_tensor))  # Make it a complex tensor
    # Apply the inverse Fourier Transform
    ifft_result = torch.fft.ifft2(psf_complex)
    # Take the magnitude of the inverse Fourier Transform result
    magnitude_ifft_result = torch.abs(ifft_result)
    # Apply fftshift to the magnitude result
    magnitude_shifted = torch.fft.fftshift(magnitude_ifft_result, dim=(-2, -1))
    batch_max = magnitude_shifted.view(magnitude_shifted.size(0), -1).max(dim=1, keepdim=True)[0]
    # Reshape to be able to broadcast across the original tensor
    batch_max = batch_max.view(-1, 1, 1, 1)
    # Normalize each batch independently
    magnitude_shifted_normalized = magnitude_shifted / batch_max

    return magnitude_shifted_normalized

def torch_ifft2_angle(psf_tensor):
    '''
    Perform the inverse Fourier Transform on a Point Spread Function (PSF) tensor using PyTorch.
    psf_tensor (torch.Tensor): The Point Spread Function (PSF) tensor (batch_size, channels, height, width).
    '''
    psf_complex = torch.complex(psf_tensor, torch.zeros_like(psf_tensor))  # Make it a complex tensor
    # Apply the inverse Fourier Transform
    ifft_result = torch.fft.ifft2(psf_complex)
    # Take the angle of the inverse Fourier Transform result
    angle_ifft_result = torch.angle(ifft_result)

    # Apply fftshift to the magnitude result
    angle_shifted = torch.fft.fftshift(angle_ifft_result, dim=(-2, -1))

    return angle_shifted

def central_crop(psf, crop_size=64, original_size=256):
    """
    Crop the central region of a 4D tensor.

    Parameters:
        psf (torch.Tensor): The input tensor (batch_size, channels, height, width).
        crop_size (int): The size of the central crop.
        original_size (int): The original size of the tensor.
    """
    
    
    # Calculate the start and end indices for cropping
    start_x = (original_size - crop_size) // 2
    end_x = start_x + crop_size
    
    start_y = (original_size - crop_size) // 2
    end_y = start_y + crop_size
    
    # Crop the central 64x64 region
    cropped_psf = psf[:, :, start_x:end_x, start_y:end_y]
    
    return cropped_psf

def patchify_blurred_img(blurred_img, patch_size=64, shuffle=False):
    """
    Patchifies a tensor into patches of the specified size and optionally shuffles the patches.
    
    Parameters:
    blurred_img (torch.Tensor): Input tensor of shape (B, 1, H, W), where H and W are divisible by patch_size.
    patch_size (int): The size of the patches (e.g., 64 or 128).
    shuffle (bool): If True, shuffles the order of the patches.
    
    Returns:
    torch.Tensor: Patchified tensor of shape (B, num_patches, patch_size, patch_size).
    """
    # Check that the height and width are divisible by the patch_size
    H, W = blurred_img.shape[2], blurred_img.shape[3]
    assert H % patch_size == 0 and W % patch_size == 0, \
        "Height and width of the input tensor must be divisible by the patch size"
    
    # Calculate the number of patches
    num_patches = (H // patch_size) * (W // patch_size)
    
    # Unfold the input tensor into patches
    patches = F.unfold(blurred_img, kernel_size=patch_size, stride=patch_size)
    
    # Reshape the patches to (B, num_patches, patch_size, patch_size)
    B = blurred_img.shape[0]
    patches = patches.view(B, num_patches, patch_size, patch_size)
    
    # If shuffle is True, shuffle the patches along the 2nd dimension
    if shuffle:
        perm = torch.randperm(num_patches)
        patches = patches[:, perm, :, :]
    
    return patches

def normalize_phase(phase_mask, range_type='0_to_2pi'):
    """
    Normalize the phase mask to the specified range.
    
    Parameters:
        phase_mask (ndarray): The phase mask to be normalized.
        range_type (str): The normalization range type ('0_to_2pi' or '-pi_to_pi').
        
    Returns:
        ndarray: The normalized phase mask.
    """
    if range_type == '0_to_2pi':
        return np.mod(phase_mask, 2 * np.pi)
    elif range_type == '-pi_to_pi':
        return (phase_mask + np.pi) % (2 * np.pi) - np.pi
    else:
        raise ValueError("Invalid range_type. Use '0_to_2pi' or '-pi_to_pi'.")

def add_gaussian_noise(tensor, std_range=(0.0, 0.08)):
    """
    Adds Gaussian noise to the input tensor within the specified standard deviation range.
    Assumes the input tensor has values scaled between 0 and 1.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, 1, 256, 256) scaled between 0 and 1.
        std_range (tuple): Tuple specifying the range of standard deviation (min, max).
                           Default range is (0.0, 0.08).

    Returns:
        torch.Tensor: Tensor with added Gaussian noise, values clamped to [0, 1].
    """
    # Randomly sample a standard deviation for each batch
    std = torch.empty(tensor.size(0), 1, 1, 1).uniform_(*std_range).to(tensor.device)
    
    # Generate Gaussian noise scaled by the sampled std
    noise = torch.randn_like(tensor) * std
    
    # Add noise to the input tensor
    noisy_tensor = tensor + noise
    
    # Clamp values to ensure they remain within [0, 1]
    return torch.clamp(noisy_tensor, 0, 1)
    

    
