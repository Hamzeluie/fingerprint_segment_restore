import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import cv2
import yaml
from models.postproccess.preprocessing.preprocess import preprocess_general
from models.postproccess.reconstruction.adjust_fitting import draw_whole_img, thick


# param_yaml_file = sys.argv[1]
param_yaml_file = "/home/naserwin/hamze/fingerprint_segment_restore/params.yaml"
params = yaml.safe_load(open(param_yaml_file))["postproccess"]
input_dir = params["input_dir"]
# input_dir = os.path.join(path_root, "results/generative")
assert os.path.isdir(input_dir), "check inputdir"
output_dir = os.path.join(path_root, "results/postproccess")
assert os.path.isdir(output_dir), "check outputdir"
png_path = [i for i in Path(input_dir).glob("**/*.png")]
jpg_path = [i for i in Path(input_dir).glob("**/*.jpg")]
jpeg_path = [i for i in Path(input_dir).glob("**/*.jpeg")]
img_list_path = png_path + jpg_path + jpeg_path
for path in img_list_path:
    img_path = path.as_posix()
    img_name = path.name.split(".")[0]
    img_broad, img_normal, img_enhance, thin_img = preprocess_general(img_path)
    cv2.imwrite(os.path.join(output_dir, img_name + "_board.png"), img_broad)
    cv2.imwrite(os.path.join(output_dir, img_name + "_norm.png"), img_normal)
    cv2.imwrite(os.path.join(output_dir, img_name + "_enhance.png"), img_enhance)
    thin_path = os.path.join(output_dir, img_name + "_thin.png")
    cv2.imwrite(thin_path, thin_img)
    background, new_img = draw_whole_img(thin_path)
    reconstructed = thick(new_img)
    # cv2.imwrite(os.path.join(output_dir, img_name + "_new.png"), new_img)
    # cv2.imwrite(os.path.join(output_dir, img_name + "_back.png"), background)
    cv2.imwrite(os.path.join(output_dir, img_name + "_reconstruct.png"), reconstructed)
"""
dvc stage add -n postproccess \
              -d results/generative \
              -d src/postproccess.py \
              -p postproccess.input_dir \
              -o results/postproccess \
              python src/postproccess.py
              
"""