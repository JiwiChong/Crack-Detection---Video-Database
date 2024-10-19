import os
import torch
import numpy as np
import glob
import argparse
import pickle
import onnx
import onnxruntime
import onnxruntime as onnx_rt

from pathlib import Path
from urllib.error import URLError
from tqdm import tqdm
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (CONFIG_DIR, FONT, LOGGER, check_font, check_requirements, clip_coords, increment_path,
                           is_ascii, xywh2xyxy, xyxy2xywh)


def converter_to_onnx(args):
    weights = os.path.join(args.main_dir, 'yolov7\\seg\\runs\\train-seg\\custom3\\weights\\best.pt')
    data= os.path.join(args.main_dir, 'smilab_data\\frames_in_image\\data.yaml')

    device = '0' if torch.cuda.is_available() else 'cpu'
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)

    im = pickle.load(open(os.path.join(args.main_dir, '.\\sample_input_onnx_test\\sample_input.pickle'), 'rb'))

    torch.onnx.export(model,
                  im,
                  os.path.join(args.main_dir, '.\\sample_yolov7_onnx_model\\yolov7_onnx.onnx'),
                  export_params=True,
                  verbose=False,              # Print verbose output
                  input_names=['input'],     # Names for input tensor
                  output_names=['output'])
    
    yolov7_onnx_model = onnx.load(os.path.join(args.main_dir, '.\\yolov7_onnx_model\\yolov7_onnx.onnx'))

    # Create an ONNX runtime session
    ort_session = onnxruntime.InferenceSession(os.path.join(args.main_dir, 'yolov7_onnx_model\\yolov7_onnx.onnx'), providers=['CPUExecutionProvider'])
    onnx.checker.check_model(yolov7_onnx_model)
    inputs = {"input": im.cpu().numpy()}
    onnx_outputs = ort_session.run(None, inputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--main_dir', type=int, help='Main directory of this repository')
    args = parser.parse_args()
    converter_to_onnx()

# command:
# python model_to_onnx_converter.py --main_dir (main directory)