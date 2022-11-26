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
[content](demo_images/yuqi.png)

#### Mask Input Image
[mask](demo_images/mask_yuqi.png)

#### Video
[video](output_videos/yuqi_zoom_in.gif)