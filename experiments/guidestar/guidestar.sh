python train_optical_experiment.py \
    --gpu 6 \
    --batch_size 256 \
    --lr 2e-5 \
    --name guidestar_PhaseNet \
    --psf_dir_train /path/to/train/psf \
    --phase_dir_train /path/to/train/phase \
    --psf_dir_valid /path/to/valid/psf \
    --phase_dir_valid /path/to/valid/phase