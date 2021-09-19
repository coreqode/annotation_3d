import cv2
import argparse
import pickle
import numpy as np
import os
from termcolor import colored, cprint
import shutil

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name',required = True,  type=str)
    parser.add_argument('--gender',required = True,  type=str)
    opts = parser.parse_args()
    return opts

def get_info(abs_path, seq_name, gender):

    ref_video_filepath = os.path.join(abs_path, 'data/to_annotate', seq_name, 'tcmr_output.mp4')
    tcmr_pkl_filepath = os.path.join(abs_path, 'data/to_annotate', seq_name, 'tcmr_output.pkl')
    blend_file_path = os.path.join(abs_path, 'data/to_annotate', seq_name, f'annotate/{seq_name}.blend')

    cprint(f'Parameter path is : {tcmr_pkl_filepath}', 'green')
    cprint(f'Video path is : {ref_video_filepath}', 'green')
    cprint(f'Gender is : {gender}', 'green')

    cap = cv2.VideoCapture(ref_video_filepath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()

    d = {'height': frame.shape[0],
         'width': frame.shape[1],
         'num_frame': frame_count,
         'tcmr_pkl_filepath': tcmr_pkl_filepath,
         'ref_video_filepath': ref_video_filepath,
         'gender': gender,
         'blend_file': blend_file_path,
         }
    with open('./temp.pkl', 'wb') as fi:
        pickle.dump( d, fi)

def prepare_annotation_folder(abs_path, seq_name):
    path = os.path.join(abs_path, 'data/to_annotate', seq_name, 'annotate')
    os.makedirs(path, exist_ok = True)

    orig_blend_path = os.path.join(abs_path, 'data/blend_files/starter.blend')
    output_blend_path = os.path.join(abs_path, 'data/to_annotate', seq_name, 'annotate', f'{seq_name}.blend')
    shutil.copy(orig_blend_path, output_blend_path)

if __name__ == '__main__':
    opts = get_parser()

    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    prepare_annotation_folder(abs_path, opts.seq_name)
    get_info(abs_path, opts.seq_name, opts.gender)

    ## Debugging the bounding boxes
    # import joblib
    # d = joblib.load('./data/tcmr_output.pkl')
    # d = d[1]
    # bbox = d['bboxes']
    # cap = cv2.VideoCapture(opts.filepath)

    # for i in range(bbox.shape[0]):
    #     ret, frame = cap.read()
    #     bx = bbox[i]


    #     cx, cy, w, h = bx[0], bx[1], bx[2], bx[3]

    #     import sys
    #     sys.exit()

    #     w = w / 1.2

    #     xmin = int(cx - w/2)
    #     ymin = int(cy - h/2)
    #     xmax = int(cx + w/2)
    #     ymax = int(cy + h/2)

    #     # print(xmin, ymin, xmax, ymax)

    #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    #     cv2.imshow('', frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
