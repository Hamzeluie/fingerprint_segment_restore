from pathlib import Path
import os
import sys
parent = Path(__file__).parents
path_root = parent[1]
sys.path.append(str(path_root))
from argparse import Namespace
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import yaml

from models.preproccess.utils.data_loading import BasicDataset
from models.preproccess.unet import UNet
from models.preproccess.utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def setParameters(params):
    data = {
        "model": params["model_path"],
        "input": params["input_dir"],
        "output": params["output_dir"] if os.path.isdir(params["output_dir"]) else f"{path_root}/results/preproccess",
        "viz": True,
        "no-save": False,
        "mask_threshold": 0.5,
        "scale": 0.5,
        "bilinear": False,
        "classes": 2
        }
        
    return data

def get_output_filenames(output, infiles):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return [os.path.join(output, "OUT_" + i.split("/")[-1]) for i in infiles]
    # return list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    # param_yaml_file = sys.argv[1]
    param_yaml_file = "/home/naserwin/hamze/fingerprint_segment_restore/params.yaml"
    params = yaml.safe_load(open(param_yaml_file))["preproccess"]
    data = setParameters(params)
    args = Namespace(**data)
    assert os.path.isfile(args.model), f"model path {args.model} is not valid."
    assert os.path.isdir(args.input), f"input_dir {args.input} is not valid"
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isdir(args.input):
        in_files = [i.as_posix() for i in Path(args.input).glob("*")]
    elif os.path.isfile(args.input):
        args.input = [args.input]
        in_files = args.input
    else:
        raise print("Oops!  input_dir is not valid. it should be path of an image file or a directory contain image files.")
        
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    out_files = get_output_filenames(args.output, in_files)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
"""
dvc stage add -n preproccess \
              -d src/preproccess.py \
              -p preproccess.model_path,preproccess.input_dir,preproccess.output_dir \
              -o results/preproccess \
              python src/preproccess.py params.yaml
"""
