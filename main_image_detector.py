import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from PIL import Image
# yolov7.seg.
os.chdir('C:/Users/jwkor/Documents/UNM/database_paper/Paper_ali_kaveh/Paper_ali_kaveh/yolov7/seg')
from yolov7.seg.models.common import DetectMultiBackend
from yolov7.seg.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov7.seg.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov7.seg.utils.plots import Annotator, colors, save_one_box
from yolov7.seg.utils.torch_utils import select_device, smart_inference_mode
from yolov7.seg.utils.segment.general import process_mask, scale_masks
from yolov7.seg.utils.segment.plots import plot_masks


def config():
    parser = argparse.ArgumentParser(description='Crack detection with Yolov7')
    parser.add_argument('--source', type=str, help='input video')
    parser.add_argument('--weights', type=str, default='C:\\Users\\jwkor\\Documents\\UNM\\database_paper\\Paper_ali_kaveh\\Paper_ali_kaveh\\yolov7\\seg\\runs\\train-seg\\custom3\\weights\\best.pt',help='number of device')
    parser.add_argument('--data', type=str, default='C:\\Users\\jwkor\\Documents\\UNM\\database_paper\\Paper_ali_kaveh\\Paper_ali_kaveh\\smilab_data\\frames_in_image\\coco128.yaml', help='sequence length')
    parser.add_argument('--project', type=str, default='C:\\Users\\jwkor\\Documents\\UNM\\database_paper\\Paper_ali_kaveh\\Paper_ali_kaveh\\yolov7\\seg\\runs\\detect\\', help='sequence length')
    parser.add_argument('--device', type=str, help='device used: cpu or cuda:0')
    parser.add_argument('--output_image', type=str, help='dir and name of output image')
    parser.add_argument('--output_video', type=str, help='dir and name of output video')
    parser.add_argument('--exp_num', type=int, help='Number of experiment')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize or not')
    parser.add_argument('--augment', type=bool, default=False, help='augment or not')
    parser.add_argument('--save_crop', type=bool, default=False, help='Number of experiment')
    parser.add_argument('--source_from_db', type=bool, help='Is source coming from DB?')
    parser.add_argument('--nosave', type=bool, default=False, help='save or no save')
    parser.add_argument('--dnn', type=bool, default=False, help='dnn or not')
    parser.add_argument('--half', type=bool, default=False, help='half or not')
    parser.add_argument('--exist_ok', type=bool, default=False, help='exist_ok or not')
    parser.add_argument('--save_txt', type=bool, default=False, help='save_txt or not')
    parser.add_argument('--view_img', type=bool, default=True)
    parser.add_argument('--save_conf', type=bool, default=False)
    parser.add_argument('--hide_labels', type=bool, default=False)
    parser.add_argument('--hide_conf', type=bool, default=False)
    parser.add_argument('--agnostic_nms', type=bool, default=False)
    parser.add_argument('--conf_thres', type=float, default=0.25, help='conf threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='iou threshold')
    parser.add_argument('--max_det', type=float, default=1000, help='max det')
    
    # parser.add_argument('--run_num', type=int,  help='run number')
    args = parser.parse_args()
    return args

def run(args):
    source = args.source
    save_img = not args.nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or args.source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    save_dir = os.path.join(args.project, 'exp{}'.format(args.exp_num))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print('Saving directory already exists!')

    device = select_device(0)
    model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=args.half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size([360,480], s=stride)  # check image size

    bs, dataset = 1, LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    vid_path, vid_writer = [None] * bs, [None] * bs

    classes = None
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    info = {}
    im0_ = []
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if args.visualize else False
            pred, out = model(im, augment=args.augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes, args.agnostic_nms, max_det=args.max_det,nm=32)
        
        det = pred[0]
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = str(save_dir +'/'+ p.name)  # im.jpg
        txt_path = str(save_dir +'/' +'labels'+ '/'+ p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if args.save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=3, example=str(names)) # pil is False 
        if len(det):
            masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            mcolors = [colors(int(cls), True) for cls in det[:, 5]]
            # # print('mccolors:', mcolors)
            im_masks = plot_masks(im[0], masks, mcolors)  # image with masks shape(imh,imw,3)
            annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w

            for *xyxy, conf, cls in reversed(det):
                if save_img or args.view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
     
            
        # Stream results
        im0 = annotator.result()
        # info['im0'] = im0
        info['masks'] = masks
        info['im_masks'] = im_masks

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                im0_.append(im0)
                cv2.imwrite(save_path, im0)

    
    masks = np.moveaxis(info['masks'].cpu().numpy(), 0, -1)
    pts = np.argwhere(masks == 1)
    segment_pts = np.array([[pts[i][1], pts[i][0], pts[i][2]] for i in range(len(pts))])
    colored_image = info['im_masks']
    for p in list(segment_pts): 
        colored_image_ = cv2.circle(colored_image, (p[0],p[1]), radius=0, color=(0, 255, 0), thickness=1)
    crack_coordinates = ','.join(str(x) for x in [[list(c) for c in list(np.array(segment_pts)[:,:2])]])
    image_to_save = Image.fromarray(colored_image_)
    image_to_save.save(os.path.join(args.project, 'exp{}'.format(args.exp_num), "crack_points.jpg"))

if __name__ == '__main__':
    args = config()
    run(args)

# command
# python main_image_detector.py --source C:/Users/jwkor/Documents/UNM/database_paper/Paper_ali_kaveh/Paper_ali_kaveh/images/crack1.jpg --device cuda:0 --output_image C:/Users/jwkor/Documents/UNM/database_paper/Paper_ali_kaveh/Paper_ali_kaveh/yolov7/seg/runs/detect/exp2/out_crack1.jpg --exp_num 2
'''
python main_image.py 
--source 
(directory and name of source image) 
--device cuda:0
--output_image (directory and name of output image)
--run_num (number of run)

'''