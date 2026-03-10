python train_kernel_estimation_dataset_patched.py \
    --gpu 6 \
    --batch_size 256 \
    --lr 2e-4 \
    --name guidestar_free_PSFNet \
    --psf_dir_train /path/to/train/psf \
    --phase_dir_train /path/to/train/phase \
    --psf_dir_valid /path/to/valid/psf \
    --phase_dir_valid /path/to/valid/phase

python train_joint_estimation_dataset_patched.py \
    --gpu 6 \
    --batch_size 200 \
    --lr 2e-5 \
    --name guidestar_free_Joint \
    --psf_dir_train /path/to/train/psf \
    --phase_dir_train /path/to/train/phase \
    --psf_dir_valid /path/to/valid/psf \
    --phase_dir_valid /path/to/valid/phase \
    --checkpoint_path_phase /path/to/PhaseNet/checkpoint \
    --checkpoint_path_psf /path/to/PSFNet/checkpoint