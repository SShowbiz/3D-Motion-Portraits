import argparse


def read_arguments():
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser)
    args = parser.parse_args()

    print_options(args, parser)

    return args


def add_all_arguments(parser):
    parser.add_argument("--content_path", type=str, default="./face.jpg", help="Image resolution")
    parser.add_argument("--mask_path", type=str, default="./background.jpg", help="Image resolution")
    parser.add_argument("--output_path", type=str, default="output_images", help="output image path")
    parser.add_argument("--text", type=str, default="Fire", help="Image resolution")
    parser.add_argument("--lambda_tv", type=float, default=2e-3, help="total variation loss parameter")
    parser.add_argument("--lambda_patch", type=float, default=9000, help="PatchCLIP loss parameter")
    parser.add_argument("--lambda_dir", type=float, default=500, help="directional loss parameter")
    parser.add_argument("--lambda_c", type=float, default=150, help="content loss parameter")
    parser.add_argument("--crop_size", type=int, default=128, help="cropped image size")
    parser.add_argument("--num_crops", type=int, default=64, help="number of patches")
    parser.add_argument("--img_width", type=int, default=472, help="size of images")
    parser.add_argument("--img_height", type=int, default=680, help="size of images")
    parser.add_argument("--max_step", type=int, default=200, help="Number of domains")
    parser.add_argument("--lr_stylize", type=float, default=5e-4, help="Number of domains")
    parser.add_argument("--thresh", type=float, default=0.7, help="Number of domains")
    parser.add_argument("--num_output", type=int, default=1, help="Number of output image")

    return parser


def print_options(opt, parser):
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        default = parser.get_default(k)
        if v != default:
            comment = "\t[default: %s]" % str(default)
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    print(message)
