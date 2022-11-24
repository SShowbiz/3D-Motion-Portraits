from PIL import Image

import os
import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models

import StyleNet
import utils
import config
import clip
import torch.nn.functional as F
from template import imagenet_templates

from PIL import Image
from torchvision import utils as vutils
from torchvision.transforms.functional import adjust_contrast
from tqdm import tqdm

args = config.read_arguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VGG = models.vgg19(pretrained=True).features
VGG.to(device)

for parameter in VGG.parameters():
    parameter.requires_grad_(False)


def img_denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = image * std + mean
    return image


def img_normalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize(image, device):
    image = F.interpolate(image, size=224, mode="bicubic")
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


content_path = args.content_path
mask_path = args.mask_path
content_image = utils.load_image2(content_path)
mask_image = utils.load_image2(mask_path)
content_name = content_path.split(".")[-2].split("/")[-1]

content_image = content_image.to(device)
mask_image = mask_image.to(device)

input_image = content_image * mask_image

content_features = utils.get_features(img_normalize(input_image), VGG)

target = input_image.clone().requires_grad_(True).to(device)

style_net = StyleNet.UNet()
style_net.to(device)

style_weights = {"conv1_1": 0.1, "conv2_1": 0.2, "conv3_1": 0.4, "conv4_1": 0.8, "conv5_1": 1.6}

content_weight = args.lambda_c

show_every = 100
optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
steps = args.max_step

content_loss_epoch = []
style_loss_epoch = []
total_loss_epoch = []

output_image = input_image

m_cont = torch.mean(input_image, dim=(2, 3), keepdim=False).squeeze(0)
m_cont = [m_cont[0].item(), m_cont[1].item(), m_cont[2].item()]

cropper = transforms.Compose([transforms.RandomCrop(args.crop_size)])
augment = transforms.Compose([transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5), transforms.Resize(224)])
device = "cuda"
clip_model, preprocess = clip.load("ViT-B/32", device, jit=False)

prompt = args.text

source = "a Photo"

with torch.no_grad():
    template_text = compose_text_with_templates(prompt, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)
    source_features = clip_model.encode_image(clip_normalize(input_image, device))
    source_features /= source_features.clone().norm(dim=-1, keepdim=True)

num_crops = args.num_crops
for epoch in tqdm(range(0, steps + 1)):

    scheduler.step()
    target = style_net(input_image, use_sigmoid=True).to(device)
    target.requires_grad_(True)

    target = target * mask_image

    target_features = utils.get_features(img_normalize(target), VGG)

    content_loss = 0

    content_loss += torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)
    content_loss += torch.mean((target_features["conv5_2"] - content_features["conv5_2"]) ** 2)

    loss_patch = 0
    img_proc = []
    for n in range(num_crops):
        target_crop = cropper(target)
        target_crop = augment(target_crop)
        img_proc.append(target_crop)

    img_proc = torch.cat(img_proc, dim=0)
    img_aug = img_proc

    image_features = clip_model.encode_image(clip_normalize(img_aug, device))
    image_features /= image_features.clone().norm(dim=-1, keepdim=True)

    img_direction = image_features - source_features
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = 1 - torch.cosine_similarity(img_direction, text_direction, dim=1)
    loss_temp[loss_temp < args.thresh] = 0
    loss_patch += loss_temp.mean()

    glob_features = clip_model.encode_image(clip_normalize(target, device))
    glob_features /= glob_features.clone().norm(dim=-1, keepdim=True)

    glob_direction = glob_features - source_features
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

    loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

    reg_tv = args.lambda_tv * get_image_prior_losses(target)

    total_loss = args.lambda_patch * loss_patch + content_weight * content_loss + reg_tv + args.lambda_dir * loss_glob
    total_loss_epoch.append(total_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # if epoch % 20 == 0:
    #     print("After %d criterions:" % epoch)
    #     print("Total loss: ", total_loss.item())
    #     print("Content loss: ", content_loss.item())
    #     print("patch loss: ", loss_patch.item())
    #     print("dir loss: ", loss_glob.item())
    #     print("TV loss: ", reg_tv.item())

    if epoch % 50 == 0:
        if not os.path.exists("output_images"):
            os.makedirs("output_images")
        out_path = "output_images/" + content_name + "_" + prompt.replace(" ", "_") + ".jpg"
        output_image = target.clone()
        output_image = torch.clamp(output_image, 0, 1)
        output_image = adjust_contrast(output_image, 1.5)
        output_image = output_image + (1 - mask_image) * content_image
        vutils.save_image(output_image, out_path, nrow=1, normalize=True)
