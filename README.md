## Text-Driven 3D Motion Portraits

### Environment Setup
```
conda env create --file environment.yaml
```

### CLIPstyler Demo
```
python CLIPstyler/stylize.py --content_path demo_images/yuqi.png --mask_path demo_images/mask_yuqi.png --text "cherry blossom"
```