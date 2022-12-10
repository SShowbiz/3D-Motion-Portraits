from CLIPstyler.stylize import stylize
from Moments3D.momentize import render as momentize
from OhMyFace.run import edit_facial_expression as generate_near_duplicate
import configargparse


def add_all_arguments(parser):
    # stylize arguments
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
    parser.add_argument(
        "--config", default="Moments3D/configs/render.txt", is_config_file=True, help="config file path"
    )
    # momentize arguments
    parser.add_argument("--rootdir", type=str, default="./", help="the path to the project root directory.")
    parser.add_argument("--expname", type=str, default="exp", help="experiment name")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 8)"
    )
    parser.add_argument("--distributed", action="store_true", help="if use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="rank for distributed training")
    parser.add_argument("--eval_mode", action="store_true", help="if in eval mode")
    parser.add_argument("--train_dataset", type=str, default="vimeo", help="the training dataset")
    parser.add_argument(
        "--dataset_weights",
        nargs="+",
        type=float,
        default=[],
        help="the weights for training datasets, used when multiple datasets are used.",
    )
    parser.add_argument("--eval_dataset", type=str, default="vimeo", help="the dataset to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size, currently only support 1")
    parser.add_argument("--feature_dim", type=int, default=32, help="the dimension of the extracted features")
    parser.add_argument("--use_inpainting_mask_for_feature", action="store_true")
    parser.add_argument("--inpainting", action="store_true", help="if do inpainting")
    parser.add_argument("--train_raft", action="store_true", help="if train raft")
    parser.add_argument("--boundary_crop_ratio", type=float, default=0, help="crop the image before computing loss")
    parser.add_argument("--vary_pts_radius", action="store_true", help="if vary point radius as augmentation")
    parser.add_argument("--adaptive_pts_radius", action="store_true", help="if use adaptive point radius")
    parser.add_argument("--use_mask_for_decoding", action="store_true", help="if use mask for decoding")
    parser.add_argument(
        "--use_depth_for_feature", action="store_true", help="if use depth map when extracting features"
    )
    parser.add_argument("--use_depth_for_decoding", action="store_true", help="if use depth map when decoding")
    parser.add_argument("--point_radius", type=float, default=1.5, help="point radius for rasterization")
    parser.add_argument("--input_dir", type=str, default="", help="input folder that contains a pair of images")
    parser.add_argument(
        "--visualize_rgbda_layers", action="store_true", help="if visualize rgbda layers, save in out dir"
    )
    parser.add_argument("--n_iters", type=int, default=250000, help="num of iterations")
    parser.add_argument("--lr_moments", type=float, default=3e-4, help="learning rate for feature extractor")
    parser.add_argument("--lr_raft", type=float, default=5e-6, help="learning rate for raft")
    parser.add_argument(
        "--lrate_decay_factor",
        type=float,
        default=0.5,
        help="decay learning rate by a factor every specified number of steps",
    )
    parser.add_argument(
        "--lrate_decay_steps",
        type=int,
        default=50000,
        help="decay learning rate by a factor every specified number of steps",
    )
    parser.add_argument("--loss_mode", type=str, default="lpips", help="the loss function to use")
    parser.add_argument(
        "--ckpt_path", type=str, default="", help="specific weights npy file to reload for coarse network"
    )
    parser.add_argument("--no_reload", action="store_true", help="do not reload weights from saved ckpt")
    parser.add_argument("--no_load_opt", action="store_true", help="do not load optimizer when reloading")
    parser.add_argument("--no_load_scheduler", action="store_true", help="do not load scheduler when reloading")
    parser.add_argument("--i_print", type=int, default=100, help="frequency of console printout and metric loggin")
    parser.add_argument("--i_img", type=int, default=500, help="frequency of tensorboard image logging")
    parser.add_argument("--i_weights", type=int, default=10000, help="frequency of weight ckpt saving")
    parser.add_argument(
        "--video_path", type=int, default=1, choices=[_ for _ in range(3)], help="video path(ex. zoom-in)"
    )

    parser.add_argument('--facial_input_dir', default='input.jpg', type=str)
    # parser.add_argument('--facial_output_dir', default='output.jpg', type=str)
    parser.add_argument('--beta', default = 0.15, type=float) #min_value=0.08, max_value=0.3, value=0.15, step=0.01)
    parser.add_argument('--alpha', default = 1.9, type=float) #min_value=-10.0, max_value=10.0, value=4.1, step=0.1)
    parser.add_argument('--gamma', default = 6, type=int) #min_value=2, max_value=10, value=6, step=1)
    parser.add_argument('--data_type', default = 'face', type=str) #['face', 'cat']
    parser.add_argument('--neutral', default = 'face')
    parser.add_argument('--target', default = 'face with smile')
    parser.add_argument('--weight_dir', default = 'OhMyFace/src/weights')

    parser.add_argument('--img', dest='img', nargs=2, default=['output.jpg', 'input_aligned.jpg'])
    parser.add_argument('--exp', default=1, type=int)
    parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
    parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
    parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
    parser.add_argument('--model', dest='modelDir', type=str, default='OhMyFace/src/weights', help='directory with trained model files')
    return parser


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser = add_all_arguments(parser)

    args = parser.parse_args()
    args.input_dir = args.output_path

    # stylize
    output = stylize(args)
    args.facial_input_dir = output
    
    # generate near-duplicated, facial expressed image 
    generate_near_duplicate(args)
    
    # momentize
    momentize(args)
