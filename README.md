## Text-Driven 3D Motion Portraits

### Environment Setup
```shell
conda env create --file environment.yml
```

### Stylize + Momentize Demo
```shell
python main.py --content_path demo_images/yuqi.png --mask_path demo_images/mask_yuqi.png --output_path yuqi --text "cherry blossom"
```

### Example Result

#### Content Input Image
<img src="demo_images/yuqi.png" width=300>

#### Mask Input Image
<img src="demo_images/mask_yuqi.png" width=300>

#### Stylized Video (Text: Cherry Blossom)
<img src="output_videos/yuqi_zoom_in.gif" width=300>
