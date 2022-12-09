from .global_singleimg_infer import global_transfer
import torch
import cv2

def inverse(args):
    latent = torch.load("tmp_latent.pt")
    beta = args.beta
    alpha = args.alpha

    result = global_transfer(latent.cpu().detach().numpy(), args.data_type, neutral= args.neutral, target = args.target, beta = beta, alpha = alpha)
    cv2.imwrite("output.jpg",result[:,:,::-1])

