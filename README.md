# Text-Driven 3D Motion Portraits

## Environment Setup

This repository environment is based on [Anaconda3](https://www.anaconda.com/products/distribution).

`Method 1`: YML file Installation

```shell
$ conda env create --file environment.yml
```

If the above command not work, try to follow `Method 2`

`Method 2`:  Manual Installation

```shell
$ conda create -n 3DMotion python=3.6

$ pip install tensorflow-gpu==1.15.2
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install -c fvcore -c iopath -c conda-forge fvcore iopath
$ conda install pytorch3d -c pytorch3d

$ pip install ftfy regex tqdm
$ pip install imageio opencv-python configargparse scipy
$ pip install timm scikit-learn gdown imageio-ffmpeg dlib
$ pip install kornia==0.5.10

$ pip install git+https://github.com/openai/CLIP.git
```

## Download Pretrained Model

To use pretrained [3D-Moments](https://github.com/google-research/3d-moments) implementation, use following command.

```shell
$ cd Moments3D
$ ./download.sh
```

To use pretrained [oh-my-face](https://github.com/P2Oileen/oh-my-face) implementation, use following command.


```shell
$ cd OhMyFace
$ wget https://drive.google.com/file/d/1efFoGShtZhcd6SCxOPu3AMbKZus478au/view?usp=sharing
$ tar -zxvf ffhq.tar.gz
$ mv ffhq src/
$ wget https://drive.google.com/file/d/1bXhWOnwCTTXTz7T7zJ1iXA717tyj-n3U/view?usp=sharing
$ tar -zxvf weights-face.tar.gz
$ mv weights src/
```

## 3D Motion Portrait Generation Demo
```shell
$ python main.py --content_path demo_images/yuqi.png --mask_path demo_images/mask_yuqi.png --output_path yuqi --text 'cherry blossom' --target 'face with smile'
```

## Main Parameter Explanation

| Parameter  | Meaning |
| --- | --- | 
| `content_path`  | base image path (for 3D Motion Portrait generation)  |
| `mask_path`  | image mask path (masked region will be stylized using `text` parameter)  | 
| `output_path` | folder directory for output files generation |
| `text` | text for masked region stylization |
| `neutral` | neutral image text description (ex. face, face with hair) |
| `target` | target text for facial expression (ex. face with smile, face with blonde hair) |
| `alpha` | Strength of facial expression |
| `beta` | Stength of disentanglement of facial expression (higher `beta` changes only the given difference between `neutral` and `target`) |
| `gamma` | Strength of RIFE's sample (higher `gamma` generates more `text` based image) |

## Example Result

### Content & Mask Input Image
<p align="center">
    <img src="demo_images/yuqi.png" width=300>
    <img src="demo_images/mask_yuqi.png" width=300>
</p>

### 3D Motion Portrait Generation Result Video

<p align="center">
    <img src="output_videos/yuqi_zoom_in.gif" width=300>
</p>

Background Stylization Text: `Cherry Blossom`  
Facial Expression Text: `face with smile` 

