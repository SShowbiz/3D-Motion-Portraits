import os
import cv2
import torch
import torchvision.transforms as T
import argparse
from torch.nn import functional as F
import warnings
import time
from .train_log.RIFE_HDv3 import Model

warnings.filterwarnings("ignore")

def facial_transfer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded v3.x HD model.")
    model.eval()
    model.device()

    tik = time.time()
    print(args.img[0])
    print(args.img[1])
    img0 = cv2.imread(args.img[0]).astype('uint8')
    img1 = cv2.imread(args.img[1]).astype('uint8')
    img0 = torch.tensor(cv2.resize(img0, (1024, 1024)),dtype=torch.float32) / 255.
    img1 = torch.tensor(cv2.resize(img1, (1024, 1024)),dtype=torch.float32) / 255.
    s0 = img0.std(dim=(0,1))
    m0 = img0.mean(dim=(0,1))
    s1 = img1.std(dim=(0,1))
    m1 = img1.mean(dim=(0,1))
    img0 = m1 + (img0 - m0) * s1 / s0
    img0 = torch.clamp(img0, 0, 1)
    img0 = img0.permute(2, 0, 1)
    img1 = img1.permute(2, 0, 1)
    img0 = img0.to(device).unsqueeze(0)
    img1 = img1.to(device).unsqueeze(0)
    n, c, h, w = img0.shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img_output = model.inference(img0, img1, 0.25, gamma=args.gamma)
    output = (img_output[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
    print("rife output.")
    # cv2.imwrite('output.jpg', (img_output[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

    tok = time.time()
    print(tok-tik)

    return output