import cv2
import os
import glob
import torch
# from fastapi import FastAPI
import numpy as np
import glob
import pickle
import pymysql
import argparse
from tqdm import tqdm
from pathlib import Path
from urllib.error import URLError
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import date
import time
# Make sure to change this directory according to where you will be saving the repository!
os.chdir('C:/Users/jwkor/Documents/UNM/database_paper/Paper_ali_kaveh/Paper_ali_kaveh/yolov7/seg/')
from yolov7.seg.models.common import DetectMultiBackend
from yolov7.seg.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov7.seg.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov7.seg.utils.plots import Annotator, colors, save_one_box
from yolov7.seg.utils.segment.general import process_mask, scale_masks
from yolov7.seg.utils.segment.plots import plot_masks
from yolov7.seg.utils.torch_utils import select_device, smart_inference_mode



def config():
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--source', type=str, help='input video')
    parser.add_argument('--weights', type=str, default='C:\\Users\\jwkor\\Documents\\UNM\\database_paper\\Paper_ali_kaveh\\Paper_ali_kaveh\\yolov7\\seg\\runs\\train-seg\\custom3\\weights\\best.pt',help='number of device')
    parser.add_argument('--data', type=str, default='C:\\Users\\jwkor\\Documents\\UNM\\database_paper\\Paper_ali_kaveh\\Paper_ali_kaveh\\smilab_data\\frames_in_image\\coco128.yaml', help='sequence length')
    parser.add_argument('--device', type=str, help='device used: cpu or cuda:0')
    parser.add_argument('--output_video', type=str, help='dir and name of output video')
    parser.add_argument('--exp_num', type=int, help='Number of experiment')
    parser.add_argument('--visualize', type=bool, default=False, help='Number of experiment')
    parser.add_argument('--augment', type=bool, default=False, help='Number of experiment')
    parser.add_argument('--save_crop', type=bool, default=False, help='Number of experiment')
    parser.add_argument('--source_from_db', type=bool, default=False, help='Is source coming from DB?')
    parser.add_argument('--get_tl_br', type=bool, default=False, help='Get top left and bottom right coordinates?')
    parser.add_argument('--run_num', type=int,  help='run number')
    args = parser.parse_args()
    return args


def source_from_database():
    connection = pymysql.connect(host="127.0.0.1", port=3307, user="root", database="ali_db")
    cursor = connection.cursor()
    # SELECT * FROM `preprocessed_videos`
    cursor.execute("SELECT * FROM preprocessed_videos") 
    output = cursor.fetchall() 

    videos = [i[3] for i in output]
    return cursor, connection, videos


def run(args):
    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=args.data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size([360,480], s=stride) 
    cursor, connection, videos = source_from_database()
    source = videos[-3] if args.source_from_db else args.source
    print('Source is:', source)
    video = cv2.VideoCapture(source)

    # #Video information
    fps = video.get(cv2.CAP_PROP_FPS)  # 30 

    dataset = LoadImages(args.source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # output video dir: './runs/predict-seg/exp4/output_crack3.mp4'
    print('output video is:', args.output_video)
    output_vid = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps , (224,384)) # these dims are dependent on masks and im_masks dims!!

    save_dir = './runs/predict-seg/exp{}'.format(args.exp_num)
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    im0_list, crack_coords = [], []
    for iter_, (path, im, im0s, vid_cap, s) in tqdm(enumerate(dataset)):
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim   # add dim in index 0 to refer to batch size

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if args.visualize else False # False
            pred, out = model(im, augment=args.augment, visualize=visualize)  # pred shape: [1, 11340, 117]
            proto = out[1]  # shape: [1, 32, 96, 120]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000, nm=32)
        
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir +'/'+ p.name)  # im.jpg
            txt_path = str(save_dir +'/' +'labels'+ '/'+ p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if args.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=3, example=str(names)) # pil is False 

            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                # print('mccolors:', mcolors)
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w

                hide_labels = False
                hide_conf=False
            
                masks = np.moveaxis(masks.cpu().numpy(), 0, -1)
                pts = np.argwhere(masks == 1)
                segment_pts = np.array([[pts[i][1], pts[i][0], pts[i][2]] for i in range(len(pts))])
                # segment_pts_.append(segment_pts)
                colored_image = im_masks
                for p in list(segment_pts): 
                    colored_image_ = cv2.circle(colored_image, (p[0],p[1]), radius=0, color=(0, 255, 0), thickness=1)
                crack_coordinates = [list(c) for c in list(np.array(segment_pts)[:,:2])]
                
                output_vid.write(colored_image_)
        
        crack_coords.append(crack_coordinates)

    
    video.release() 
    output_vid.release() 
    # Closes all the frames 
    cv2.destroyAllWindows()

    current_date = '%s %s' % (date.today(), time.strftime("%H:%M:%S", time.localtime()))
    crack_coordinates = ','.join(str(x) for x in [crack_coords])
    file_coors = open('.\\runs\\output_dicts\\exp{}\\crack_coordinates.txt'.format(args.exp_num), 'w')
    file_coors.write(crack_coordinates)
    file_coors.close()


    sql_command = """insert into `postprocessed_data` (id, transformName, prevideo_id, json_points, 
                        image_location, date)
                                    
            values (%s, %s, %s, %s, %s, %s) 
        """
    cursor.execute(sql_command,(685, 'output_crack.mp4', 2, crack_coordinates, '/output_crack.mp4', current_date))
    connection.commit()



if __name__ == '__main__':
    args = config()
    run(args)




# python command: 
 
# Video
# python main_video_detector.py --source C:/Users/jwkor/Documents/UNM/database_paper/Paper_ali_kaveh/Paper_ali_kaveh/videos/crack10.mp4 --device cuda:0 --output_video C:/Users/jwkor/Documents/UNM/database_paper/Paper_ali_kaveh/Paper_ali_kaveh/yolov7/seg/runs/predict-seg/exp6/out_crack10.mp4 --exp_num 6
