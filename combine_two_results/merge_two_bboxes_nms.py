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


from nms.nms import py_nms_wrapper

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

    ret = defaultdict(list)
    # load det results in one frame
    forward_file = sys.argv[1]
    with open(forward_file, 'r') as f:
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
    
    ## load backward
    backward_file = sys.argv[2]
    with open(backward_file, 'r') as f:
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

    ## load test.output
    output = open(sys.argv[3], 'w')

    # threshold, 0.3 or 0.7
    nms = py_nms_wrapper(0.3)
    for frame_index in ret.keys():
        print (frame_index)
        # nms -> 
        dets= defaultdict(list)
        for bbox_index, item in enumerate(ret[frame_index]):
            bbox = np.hstack((item.bbox, [item.score]))
            dets[item.class_index].append(bbox)
        for class_in_frame in dets.keys():
            cls_dets = np.asarray(dets[class_in_frame])
            keep = nms(cls_dets)
            keep_bboxes = cls_dets[keep,:]
            for keep_bbox in keep_bboxes:
              outbbox = "{} {} {} {} {} {} {}\n".format(frame_index, class_in_frame, keep_bbox[-1], keep_bbox[0], keep_bbox[1], keep_bbox[2], keep_bbox[3] )
              output.write(outbbox)
        # write the output
    output.close()
          




