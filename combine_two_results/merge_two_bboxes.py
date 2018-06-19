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


def cal_IoU(bbox1, bbox2):
        # calculate real IoU
        bi = [np.max((bbox1[0], bbox2[0])), np.max((bbox1[1], bbox2[1])), np.min((bbox1[2], bbox2[2])),  np.min((bbox1[3], bbox2[3]))]
        iw = bi[2] - bi[0] + 1
        ih = bi[3] - bi[1] + 1
        if iw > 0 and ih > 0:
                ua = (bbox2[2] - bbox2[0] + 1.) * (bbox2[3] - bbox2[1] + 1.) + (bbox1[2] - bbox1[0] +1.)* (bbox1[3] - bbox1[1] + 1.) - iw * ih
                ov = iw * ih / ua
        else:
                ov = 0.
        return ov

if __name__ == "__main__":

    # load det results in one frame, results 1
    ret_forward = defaultdict(list)
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
        ret_forward[item.fid].append(item)
    print len(ret_forward.keys())
    
    ret_backward = defaultdict(list)
    ## load backward, result 2
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
        ret_backward[item.fid].append(item)
    print len(ret_backward.keys())

    ## load test.output
    output = open(sys.argv[3], 'w')

    set_ret_f = set(ret_forward.keys())
    set_ret_b = set(ret_backward.keys())
    # set forward  & backward
    ret_fb_keys = list(set_ret_f.intersection(set_ret_b))
    print (len(ret_fb_keys))

    # nms
    nms = py_nms_wrapper(0.3)
    for frame_index in ret_fb_keys:
      print (frame_index)
      # forwards bboxes
      dets_f = defaultdict(list)
      for bbox_index, item in enumerate(ret_forward[frame_index]):
         bbox = np.hstack((item.bbox, [item.score]))
         dets_f[item.class_index].append(bbox)
      # backward bboxes
      dets_b = defaultdict(list)
      for bbox_index, item in enumerate(ret_backward[frame_index]):
         bbox = np.hstack((item.bbox, [item.score]))
         dets_b[item.class_index].append(bbox)

      # intersection classes 
      class_keys_f = set(dets_f.keys())
      class_keys_b = set(dets_b.keys())
      # f only
      class_keys_f_b = list(class_keys_f - class_keys_b)
      # only classes in forward
      for class_in_frame in class_keys_f_b:
        for keep_bbox in dets_f[class_in_frame]:
          outbbox = "{} {} {} {} {} {} {}\n".format(frame_index, class_in_frame, keep_bbox[-1], keep_bbox[0], keep_bbox[1], keep_bbox[2], keep_bbox[3] )
          output.write(outbbox)

      # only classes in backward
      class_keys_b_f = list(class_keys_b - class_keys_f)
      for class_in_frame in class_keys_b_f:
        for keep_bbox in dets_b[class_in_frame]:
          outbbox = "{} {} {} {} {} {} {}\n".format(frame_index, class_in_frame, keep_bbox[-1], keep_bbox[0], keep_bbox[1], keep_bbox[2], keep_bbox[3] )
          output.write(outbbox)

      # both f and b
      class_keys_f_b = list(class_keys_f.intersection(class_keys_b))
      for class_in_frame in class_keys_f_b:
        print class_in_frame
        #  loop for two bboxes
        # print (dets_f[class_in_frame])
        # print (dets_b[class_in_frame])
        dets_class = []
        for det in dets_f[class_in_frame]:
          dets_class.append(det)
        for det in dets_b[class_in_frame]:
          dets_class.append(det)
        
        for det_f in dets_f[class_in_frame]:
          for det_b in dets_b[class_in_frame]:
            iou = cal_IoU(det_f[:4], det_b[:4])
            # print (det_f, det_b, iou)
            if iou > 0.3:
              new_bbox = (det_f + det_b)/2
              new_bbox[-1] = max(det_f[-1],det_b[-1]) + np.finfo(new_bbox[-1].dtype).eps
              # print (new_bbox)
              dets_class.append(new_bbox)
        cls_dets = np.asarray(dets_class)
        keep = nms(cls_dets)
        keep_bboxes = cls_dets[keep,:]
        for keep_bbox in keep_bboxes:
          outbbox = "{} {} {} {} {} {} {}\n".format(frame_index, class_in_frame, keep_bbox[-1], keep_bbox[0], keep_bbox[1], keep_bbox[2], keep_bbox[3] )
          output.write(outbbox)
        # write the output
        
    # set forward  - backward
    ret_f_b_keys = list(set_ret_f - set_ret_b)
    print (len(ret_f_b_keys))
    for frame_index in ret_f_b_keys:
      print (frame_index)
      # forwards bboxes
      for bbox_index, item in enumerate(ret_forward[frame_index]):
        keep_bbox = np.hstack((item.bbox, [item.score]))
        outbbox = "{} {} {} {} {} {} {}\n".format(frame_index, class_in_frame, keep_bbox[-1], keep_bbox[0], keep_bbox[1], keep_bbox[2], keep_bbox[3] )
        output.write(outbbox)
     

    # set backward - forward
    ret_b_f_keys = list(set_ret_b - set_ret_f)
    print (len(ret_b_f_keys))
    for frame_index in ret_b_f_keys:
      print (frame_index)
      # forwards bboxes
      for bbox_index, item in enumerate(ret_backward[frame_index]):
        keep_bbox = np.hstack((item.bbox, [item.score]))
        outbbox = "{} {} {} {} {} {} {}\n".format(frame_index, class_in_frame, keep_bbox[-1], keep_bbox[0], keep_bbox[1], keep_bbox[2], keep_bbox[3] )
        output.write(outbbox)
 
    output.close()
    exit()

