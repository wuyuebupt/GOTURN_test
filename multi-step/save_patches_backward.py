# train file
# video forward tracking using detection results from previous frames

import sys
import logging
import time
import os
import cv2
import numpy as np
import gc

from collections import defaultdict
from easydict import EasyDict

# parameters
# BATCH_SIZE = 1
BATCH_SIZE = 64
WIDTH = 227
HEIGHT = 227


# VID test
VID_ROOT = '/home/yue/project/vid/code/videoVisualization/myVisual/ILSVRC2015/Data/VID/val/'
ANNO_ROOT = '/home/yue/project/vid/code/videoVisualization/myVisual/ILSVRC2015/Annotations/VID/val/'
IMAGESET_ROOT = '/home/yue/project/vid/code/videoVisualization/ILSVRC/ImageSets/VID/'


fid_to_path = {}
with open(os.path.join(IMAGESET_ROOT, 'val.txt')) as f:
        lines = f.readlines()
        for line in lines:
                line = line.strip().split()
                fid_to_path[int(line[1])] = os.path.join(VID_ROOT, line[0] + '.JPEG')
length_fids = len(fid_to_path)

if __name__ == "__main__":


    # load det results in one frame
    # 
    det_file = sys.argv[1]
    ret = defaultdict(list)
    with open(det_file, 'r') as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip().split()
        item = {
                'fid': int(line[0]),
                'class_index': int(line[1]),
                'score': float(line[2]),
                'bbox': map(float, line[3:])
                }
        item = EasyDict(item)
        ret[item.fid].append(item)
    print len(ret.keys())

    ## open the output file
    out_dir = 'tmp_images/'

   # prepare data in one frame
    forward_step = -1

  
    dataset = None
    for frame_index in ret.keys():
      img_current_path = fid_to_path[frame_index]
      frame_index_next = frame_index + forward_step

      # last frame 
      if frame_index_next <= 0 :
        print ("the first frame")
        continue

      img_next_path = fid_to_path[frame_index + forward_step]
      ## check if two images are from the same video
      # print (img_current_path)
      # print (img_next_path)
      vid_current = img_current_path.split('/')[-2]
      vid_next    = img_next_path.split('/')[-2]
      if vid_current == vid_next:
        # generate two pairs of image patches for tracking 
        # target is in the middle
        # search is for the tracking 

        start_time = time.time()

        img_current = cv2.imread(img_current_path)
        img_next = cv2.imread(img_next_path)
        targetbox = np.float32([[56.75, 56.75], [56.75,170.25], [170.25, 170.25]])

        for bbox_index, item in enumerate(ret[frame_index]):
            bbox = item.bbox 
            pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
            # get warpped image
            M = cv2.getAffineTransform(pts ,targetbox)
            bbox_current = cv2.warpAffine(img_current, M, (WIDTH, HEIGHT))
            bbox_next    = cv2.warpAffine(img_next, M, (WIDTH, HEIGHT))
            basename = '%06d_%04d.jpg' % (frame_index, bbox_index)
            print basename
            save_current_path = os.path.join(out_dir, 'current', basename)
            save_next_path = os.path.join(out_dir, 'next', basename)
            # print (save_current_path)
            # print (save_next_path)
            cv2.imwrite(save_current_path, bbox_current)
            cv2.imwrite(save_next_path, bbox_next)
            

