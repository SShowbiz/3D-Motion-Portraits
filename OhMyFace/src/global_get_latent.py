from .cog_predict import get_latent_code
import cv2
import torch

def get_latent(args):
    img = cv2.imread(args.facial_input_dir)
    print(f'original_input_shape: {img.shape}')
    latent, img, img_crop, crop, quad, face_poly = get_latent_code(img, args.data_type, args.weight_dir)
    latent = latent.unsqueeze(0)

    print("print aligned image:",cv2.imwrite("input_aligned.jpg",img))
    torch.save(latent,"tmp_latent.pt")

    return img, img_crop, crop, quad, face_poly
