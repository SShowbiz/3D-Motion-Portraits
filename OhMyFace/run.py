import os
import argparse
from .src.global_get_latent import get_latent
from .src.global_run import inverse
from .src.rife.inference_img import facial_transfer
from .src.warp import compute_h_norm, warp_image
from .src.poisson_image_edit import poisson_edit
from PIL import Image
import numpy as np
import cv2

def edit_facial_expression(args):
    _, img_crop, crop, quad = get_latent(args)
    inverse(args)
    output = facial_transfer(args)

    h, w, _ = output.shape
    H = compute_h_norm(np.array([quad[0], quad[3], quad[2], quad[1]]), np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    _, igs_merge = warp_image(np.asarray(output), np.asarray(img_crop), H)
    crop_left, crop_up, crop_right, crop_bottom = crop

    h, w, _ = output.shape
    input = cv2.imread(args.facial_input_dir)
    h_input, w_input, _ = input.shape

    source = np.zeros((h_input, w_input, 3))
    source[crop_up:crop_bottom, crop_left:crop_right, :] = igs_merge

    mask = np.zeros((h_input, w_input))
    mask[crop_up+130:crop_bottom-90, crop_left+130:crop_right-130] = 1

    poisson_blend_result = poisson_edit(source, input, mask)
    
    input_path = args.facial_input_dir
    output_path = input_path.split('.')[0] + '_face_edit.png'
    cv2.imwrite(output_path, poisson_blend_result)

    os.system('rm output.jpg input_aligned.jpg')
