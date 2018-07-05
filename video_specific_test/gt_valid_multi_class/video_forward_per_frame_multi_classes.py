import os,sys
import cv2
from collections import defaultdict
from easydict import EasyDict

from vdetlib.utils.protocol import vid_proto_from_dir, frame_path_at
from vdetlib.tools.imagenet_annotation_processor import get_anno, track_by_id

import goturn_net
import tensorflow as tf

from scipy.optimize import linear_sum_assignment
import math


BATCH_SIZE = 64
WIDTH = 227
HEIGHT = 227


VID_ROOT = '/home/yue/project/vid/code/videoVisualization/myVisual/ILSVRC2015/Data/VID/val/'
ANNO_ROOT = '/home/yue/project/vid/code/videoVisualization/myVisual/ILSVRC2015/Annotations/VID/val/'
IMAGESET_ROOT = '/home/yue/project/vid/code/videoVisualization/ILSVRC/ImageSets/VID/'

CLASS_NAMES = ['background', 'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra']

PRESET_COLORS = [
  (255, 65, 54),
  (61, 153, 112),
  (0, 116, 217),
  (133, 20, 75),
  (0, 31, 63),
  (240, 18, 190),
  (1, 255, 112),
  (127, 219, 255),
  (255, 133, 27),
  (176, 176, 176)
]

## some colors
# RGB -> BGR
COLORS = {
	'green' : (0,255,127),
	'yellow': (0,215,255),
	'red':(0,0,255),
	'blue': (255,100,100),
	'gray': (85,85,85),
	'black': (0,0,0)
}
COLORS = EasyDict(COLORS)
FontColor = (0,0,0)


def track_class_at_frame(tracklet, frame_id):
    for box in tracklet:
        if box['frame'] == frame_id:
            return box['class']
    return None


def track_box_at_frame(tracklet, frame_id):
    for box in tracklet:
        if box['frame'] == frame_id:
            return box['bbox']
    return None


import numpy as np

def get_gt_thres(gt_bbox):
	# get gt_threshold -> for small objects
	gt_w = gt_bbox[2] - gt_bbox[0] + 1
	gt_h = gt_bbox[3] - gt_bbox[1] + 1
	thres = (gt_w * gt_h)/((gt_w + 10.) * (gt_h + 10.))
	gt_thr = np.min((0.5, thres))
	return gt_thr

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
	

def det_to_tracklet(ret):
	#
	track_id = 0
	for frame_index in ret.keys():
		# # print frame_index
		# # print ret[frame_index]
		if frame_index - 1 in ret.keys():
			#  get previous tracklet
			# det_previous = ret[frame_index - 1][1:2]
			det_previous = ret[frame_index - 1]
			det_current = ret[frame_index]
			# for det_index, det in enumerate(ret[frame_index]):
			# 	# print det_index, det
			# # print (det_previous)
			# # print (det_current)
			# number of previous
			num_prev = len(det_previous)
			num_curr = len(det_current)
			cost = np.zeros((num_prev, num_curr))
			# # print cost.shape
			for i in range(num_prev):
				for j in range(num_curr):
					bbox_prev = det_previous[i]
					bbox_curr = det_current[j]
					# # print bbox_prev, bbox_curr
					iou = 1 - cal_IoU(bbox_prev.bbox, bbox_curr.bbox)
					# # print iou
					cost[i,j] = iou
			# hungarian
			row_ind, col_ind = linear_sum_assignment(cost)
			# # print row_ind, col_ind

			# assign fid, 
			# and the left start a new track
			assert (len(row_ind) == len(col_ind))
			for i in range(num_curr):
				if i in col_ind:
					# find i in col_ind 
					i_ind = np.where(col_ind == i)[0][0]
					# # print i_ind
					# get previous track
					tmp_track_id = det_previous[i_ind]['track_id']
					# # print (tmp_track_id)
					det_current[i]['track_id'] = tmp_track_id
				else:
					# start a new track
					tmp_track_id = track_id
					det_current[i]['track_id'] = tmp_track_id
					track_id += 1
			
		else:
			# the first frame
			for det_index, det in enumerate(ret[frame_index]):
				# # print det_index, det
				det['track_id'] = track_id
				track_id += 1
			
		# # print ret[frame_index]

	# move ret to track
	det_tracklets = []
	for frame_index in ret.keys():
		dets = ret[frame_index]
		for det in dets:
			# # print det
			track_id_cur = det['track_id']
			track = track_by_id(det_tracklets, track_id_cur)
			if not track:
				track = {'id': track_id_cur, 'track': []}
				det_tracklets.append(track)
			track['track'].append(det)
		# # print det_tracklets
	return det_tracklets


def match_d_t(det_bbox, track_bbox):
	num_d = len(det_bbox)
	num_t = len(track_bbox)
	cost = np.zeros((num_d, num_t))
	for i in range(num_d):
		for j in range(num_t):
			bbox_d = det_bbox[i]
			bbox_t = track_bbox[j]
			# # print bbox_prev, bbox_curr
			iou = 1 - cal_IoU(bbox_d.bbox, bbox_t.bbox)
			# # print iou
			cost[i,j] = iou

	# hungarian
	row_ind, col_ind = linear_sum_assignment(cost)
	# # print row_ind, col_ind
	# assign fid, 
	assert (len(row_ind) == len(col_ind))
	assign_d = []
	for i in range(num_d):
		if i in col_ind:
			# find i in col_ind 
			i_ind = np.where(col_ind == i)[0][0]
			# # print i_ind
			assign_d.append(i_ind)
			# get previous track
			# # print (tmp_track_id)
		else:
			assign_d.append(-1)
			# start a new track
	assign_t = []
	for i in range(num_t):
		if i in row_ind:
			i_ind = np.where(row_ind == i)[0][0]
			# # print i_ind
			assign_t.append(i_ind)
		else:
			assign_t.append(-1)
	return assign_d, assign_t


def assign_res_to_gt( res_bbox, ground_truth_bboxes):
	num_res = len(res_bbox)
	num_gt = len(ground_truth_bboxes)
	cost = np.zeros((num_res, num_gt))
	for i in range(num_res):
		for j in range(num_gt):
			bbox_res = res_bbox[i]
			bbox_gt = ground_truth_bboxes[j]
			# # print bbox_prev, bbox_curr
			iou = -1* math.log(cal_IoU(bbox_res.bbox, bbox_gt)+np.finfo(float).eps)
			# iou = 1 - cal_IoU(bbox_res.bbox, bbox_gt)
			# # print iou
			cost[i,j] = iou

	# hungarian
	row_ind, col_ind = linear_sum_assignment(cost)
	# # print row_ind, col_ind
	# assign fid, 
	assert (len(row_ind) == len(col_ind))
	hard_threshold = -1 * math.log(0.1)
	assign_res = []
	# # print (hard_threshold)
	for i in range(num_res):
		if i in col_ind:
			i_ind = np.where(col_ind == i)[0][0]
			cost_i_j = cost[i_ind, i]
			# # print cost_i_j
			if cost_i_j <= hard_threshold:
				assign_res.append(i_ind)
			else:
				assign_res.append(-1)
		else:
			assign_res.append(-1)
	assign_gt = []
	for i in range(num_gt):
		if i in assign_res:
			# i_ind = np.where(assign_res == i)[0][0]
			i_ind = assign_res.index(i)
			# # print i_ind
			assign_gt.append(i_ind)
		else:
			assign_gt.append(-1)
	return assign_res, assign_gt


def split_class(ret):
	ret_class_wise = {}
	for frame_index in ret.keys():
		for det in ret[frame_index]:
			if det.class_index not in ret_class_wise.keys():
				ret_class_wise[det.class_index] = defaultdict(list)
			ret_class_wise[det.class_index][det.fid].append(det)
	# print ret_class_wise.keys()
	
	return ret_class_wise

if __name__ == '__main__':
	vid_name = sys.argv[2]
	
	# print (vid_name)
	## load vid tracks
	video_foler = VID_ROOT + vid_name
	vid = vid_proto_from_dir(video_foler, vid_name)
	# print (len(vid['frames']))
	
	# load gt
	anno_folder = ANNO_ROOT + vid_name
	annot = get_anno(anno_folder)
	# print (annot.keys())
	# print (annot['video'])
	# print (len(annot['annotations']))
	# # print (annot['annotations'][0])
	# print (len(annot['annotations'][0]))
	# print (len(annot['annotations'][1]))
	# print (len((annot['annotations'][0]['track'])))
	# print (((annot['annotations'][0]['id'])))
	# print (len((annot['annotations'][1]['track'])))
	# print (((annot['annotations'][1]['id'])))
	
	## load the detection results
	## read_submission
	
	det_file = sys.argv[1]
	
	# fid   -> image
	# video -> fid
	videos = defaultdict(list)
	fid_to_path = {}
	with open(os.path.join(IMAGESET_ROOT, 'val.txt')) as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split()
			fid_to_path[int(line[1])] = os.path.join(VID_ROOT, line[0] + '.JPEG')
			videos[os.path.dirname(line[0])].append(int(line[1]))

	# sort frames inside each video
	for k in videos:
		videos[k].sort()
	
	# find fids that are needed
	fids = videos[vid_name]
	# print (fids)
	
	# for v2 selection
	fid_min = fids[0]
	fid_max = fids[-1]
	
	ret = defaultdict(list)
	with open(det_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split()
			# v2: use order infor
			# # print (fids)
			if int(line[0]) >= fid_min and int(line[0]) <= fid_max:
				item = {
					'fid': int(line[0]),
					'class_index': int(line[1]),
					'score': float(line[2]),
					'bbox': map(float, line[3:]),
					'flag': 'd'
				}
				item = EasyDict(item)
				ret[item.fid].append(item)
				# # print (len(fids))
			if int(line[0]) > fid_max:
				break

	# print (len(ret))

	## split results into multi classes
	ret_classes = split_class(ret)
	# print ret_classes.keys()

	## det_tracklets = det_to_tracklet(ret)
	## # print (len(det_tracklets))

	## load tracking model 
	tracknet = goturn_net.TRACKNET(BATCH_SIZE, train = False)
	tracknet.build()
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	init_local = tf.local_variables_initializer()
	sess.run(init)
	sess.run(init_local)
	
	ckpt_dir = "./checkpoints"
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	ckpt = tf.train.get_checkpoint_state(ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver = tf.train.Saver()
		saver.restore(sess, ckpt.model_checkpoint_path)
	
	## match the gt with detection for each video
	## show the gt
	
	output = open(sys.argv[3], 'w')
	# remove the current images

	for ret_per_class in ret_classes.keys():
	  current_class = ret_per_class
	  print ret_per_class
	  print len(ret_classes[ret_per_class])
	  frames_ret = sorted(ret_classes[ret_per_class].keys())
	  ret_current_class = ret_classes[ret_per_class]
	  print frames_ret
	  assert (len(frames_ret) > 1)
	  if len(frames_ret) == 1:
	    # for item in ret_classes[ret_per_class][frames_ret[0]]:
	    for item in ret_current_class[frames_ret[0]]:
		outbbox = "{} {} {} {} {} {} {} d\n".format(item.fid, item.class_index, item.score, item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3])
		output.write(outbbox)
	  else:
	    ## output the results
	    # for frame_index, frame_index_abs in enumerate(frames_ret[:-1]):
	    #    print frame_index, frame_index_abs
	    for frame_index, frame in enumerate(vid['frames'][:-1]):
		print frame_index, frame
		frame_index_abs = frame_index + fid_min
		print frame_index, frame_index_abs
	# for frame_index, frame in enumerate(vid['frames']):
	# for det_tracklet in det_tracklets
	#     # print det_tracklet['id']
	#     det_tracklet_track = det_tracklet['track']
	#     for obj_index, obj in enumerate(det_tracklet_track[:-1]):
		if frame_index == 0:
			## output the first detection
			# frame_index_abs = fids[frame_index]
			# for item in ret[frame_index_abs]:
			for item in ret_current_class[frame_index_abs]:
				outbbox = "{} {} {} {} {} {} {} d\n".format(item.fid, item.class_index, item.score, item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3])
				output.write(outbbox)

		# frame_index = obj['fid']
		# frame_index_abs = fids[frame_index]

		# get frame
		# print fid_min, frame_index_abs
		# obj_index_next = obj_index + 1
		
		# frame_index = fids[frame_index]

		## ## check if two images are from the same video
		frame_index_next_abs = frame_index_abs + 1
		img_current_path = fid_to_path[frame_index_abs]
		img_next_path = fid_to_path[frame_index_next_abs]
		vid_current = img_current_path.split('/')[-2]
		vid_next    = img_next_path.split('/')[-2]
		
		# if vid_current != vid_next:
		# 	## the last frame in the video -> output
		# 	break
		# get the next frame
		imgpath = img_next_path
		imgbasename = os.path.basename(imgpath)
		imgsavepath = os.path.join('saveImgs/', imgbasename)

		# frame_next = vid['frames'][frame_index + 1]
		# frame_next = frame_index_abs + 1
		frame_relative_index_next = frame_index_abs - fid_min + 1
	        boxes = [track_box_at_frame(tracklet, frame_relative_index_next) for tracklet in [anno['track'] for anno in annot['annotations']]]
	        classes = [track_class_at_frame(tracklet, frame_relative_index_next) for tracklet in [anno['track'] for anno in annot['annotations']]]
	        # boxes = [track_box_at_frame(tracklet, frame['frame']) for tracklet in [anno['track'] for anno in annot['annotations']]]
	        # classes = [track_class_at_frame(tracklet, frame['frame']) for tracklet in [anno['track'] for anno in annot['annotations']]]
	        # boxes = [track_box_at_frame(tracklet, frame_next['frame']) for tracklet in [anno['track'] for anno in annot['annotations']]]
	        # classes = [track_class_at_frame(tracklet, frame_next['frame']) for tracklet in [anno['track'] for anno in annot['annotations']]]
		# # print (boxes)
		# # print (classes)


		# get current detects 
		# preds = ret[frame_index_abs]
		preds = ret_current_class[frame_index_abs]

		# previous frames does not have any bboxes
		# get det results 
		det_frame_index_next = ret_current_class[frame_index_next_abs]
		if len(preds) == 0:
			# check detection results
			for item in det_frame_index_next:
				outbbox = "{} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {}\n".format(item.fid, item.class_index, item.score, item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3], item.flag)
				output.write(outbbox)
			continue	


		# else do the tracknig -> 
		# preds = [obj]

		##################################### tracking

		# print (img_current_path)
		img_current = cv2.imread(img_current_path)
		img_next = cv2.imread(img_next_path)
		targetbox = np.float32([[56.75, 56.75], [56.75,170.25], [170.25, 170.25]])
		
		patches_current = []
		patches_next = []
		# print preds
		for item in preds:
		# for item in [obj]:
			bbox = item.bbox
			pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
			# get warpped image
			M = cv2.getAffineTransform(pts ,targetbox)
			bbox_current = cv2.warpAffine(img_current, M, (WIDTH, HEIGHT))
			bbox_next    = cv2.warpAffine(img_next, M, (WIDTH, HEIGHT))
			# cv2.imshow('abc',bbox_current)
			# cv2.imshow('def',bbox_next)
			# cv2.waitKey()
			patches_current.append(bbox_current)
			patches_next.append(bbox_next)
		num_bboxes = len(patches_current)
		# # print (num_bboxes)

		patches_current = np.asarray(patches_current, np.float64)
		patches_next = np.asarray(patches_next, np.float64)
		
		# patches_current_tensors = sess.run(tf.convert_to_tensor(patches_current, dtype=tf.float64))
		# patches_current_tensors = sess.run(tf.image.resize_images(patches_current_tensors,[HEIGHT,WIDTH],
		#                   method=tf.image.ResizeMethod.BILINEAR))
		# patches_next_tensors = sess.run(tf.convert_to_tensor(patches_next, dtype=tf.float64))
		# patches_next_tensors = sess.run(tf.image.resize_images(patches_next_tensors,[HEIGHT,WIDTH],
		#                   method=tf.image.ResizeMethod.BILINEAR))
		# # print (patches_current_tensors.shape)
		# fc4 = sess.run(tracknet.fc4, feed_dict={tracknet.image:patches_next_tensors, tracknet.target:patches_current_tensors})
		fc4 = sess.run(tracknet.fc4, feed_dict={tracknet.image:patches_next, tracknet.target:patches_current})
		# # print (fc4)

		next_bboxes = []
		for i in range(num_bboxes):
			x1 = (227* fc4[i][0]/10)
			y1 = (227* fc4[i][1]/10)
			x2 = (227* fc4[i][2]/10)
			y2 = (227* fc4[i][3]/10)
			next_bboxes.append([x1,y1,x2,y2])
		
		tracking_frame_index_next = []
		for next_bbox, current_bbox in zip(next_bboxes,ret_current_class[frame_index_abs]):
		# for next_bbox, current_bbox in zip(next_bboxes,frames_ret[frame_index_abs]):
		# for next_bbox, current_bbox in zip(next_bboxes, [obj]):
			bbox = current_bbox.bbox
			pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
			M_inv = cv2.getAffineTransform(targetbox , pts)
			pts_new = np.float32( [[next_bbox[0], next_bbox[1], 1], [next_bbox[2], next_bbox[3], 1]])
			# warp back the bboxes
			original_bbox = np.matmul(M_inv, pts_new.T)
			tracked_bbox = [original_bbox[0][0], original_bbox[1][0], original_bbox[0][1], original_bbox[1][1]]
			item = {
			        'fid': current_bbox.fid + 1,
			        'class_index': current_bbox.class_index,
			        'score': current_bbox.score,
			        'bbox': map(float, tracked_bbox),
				'flag': 't'
			      }
			# # print (item)
			item = EasyDict(item)
			tracking_frame_index_next.append(item)
		## 

		##################################### tracking
		# det_frame_index_next = frames_ret[frame_index_next_abs]
		# det_frame_index_next = [det_tracklet_track[obj_index_next]]


		# compare two results and update for the next frame
		# assert (len(det_frame_index_next) >= 1)
		# assert (len(tracking_frame_index_next) >= 1)
		## find the match one, started from objects
		
		## get all gt bboxes


		ground_truth_bboxes = []
		for bbox_, class_ in zip(boxes, classes):
			if bbox_ != None and class_ != None:
				if class_ == CLASS_NAMES[ current_class ]:
					ground_truth_bboxes.append(bbox_)
		# # print preds
		# # print tracking_frame_index_next
		# # print det_frame_index_next
		# # print ground_truth_bboxes
					
		### three inputs here
		### tracking results  : tracking_frame_index_next
		### detection results : det_frame_index_next
		### ground truth      : for the specific classes 

		# assign tracking to GT
		num_tracking = len(tracking_frame_index_next)
		num_gt = len(ground_truth_bboxes)
		# # print num_tracking, num_gt
		assign_tracking, assign_gt_t = assign_res_to_gt( tracking_frame_index_next, ground_truth_bboxes)
		# # print assign_tracking, assign_gt_t
		assign_det, assign_gt_d = assign_res_to_gt( det_frame_index_next, ground_truth_bboxes)
		# # print assign_det, assign_gt_d
 		# assign_d, assign_t = match_d_t(det_frame_index_next, tracking_frame_index_next)  
		# # print assign_d, assign_t

		### for every GT, get best assigned if 
		### 1. gt -> t and gt-> d 	: save t, remove d
		### 2. gt -> t and gt no-> d	: save t
		### 3. the rest d		: save d
		
		# assign t and d
		g_d_t_pairs 	= []
		g_t_matched   	= []
		d_rest		= []
		# 
		for i in range(len(ground_truth_bboxes)):
			if  assign_gt_t[i] != -1:
				if assign_gt_d[i] != -1:
					g_d_t_pairs.append([i, assign_gt_t[i], assign_gt_d[i]])
				else:
					# assign_gt_d[i] no match
					g_t_matched.append([i, assign_gt_t[i]])
		for g_d_t_pair in g_d_t_pairs:
			# # print g_d_t_pair
			d_matched = g_d_t_pair[2]
			# # print d_matched
			# mask as -2
			assign_det[d_matched] = -2
		for d_index, d_mask in enumerate(assign_det):
			if d_mask != -2:
				d_rest.append(d_index)
		# # print assign_det
		# print d_rest

		## generate the output
		d_and_t_bboxes = []
		# case 1
		for g_d_t_pair in g_d_t_pairs:
			# print g_d_t_pair
			gt_index = g_d_t_pair[0]
			t_index = g_d_t_pair[1]
			d_index = g_d_t_pair[2]
			gtbbox = ground_truth_bboxes[gt_index]
			t_bbox = tracking_frame_index_next[t_index]
			d_bbox = det_frame_index_next[d_index]
			tracking_iou 	= cal_IoU(gtbbox, t_bbox.bbox)
			det_iou 	= cal_IoU(gtbbox, d_bbox.bbox)
			# print tracking_iou, det_iou
			if tracking_iou > det_iou:
				d_and_t_bboxes.append(t_bbox)
			else:
				d_and_t_bboxes.append(d_bbox)
		# case 2
		for g_t_index in g_t_matched:
			t_index = g_t_index[1]
			t_bbox = tracking_frame_index_next[t_index]	
			d_and_t_bboxes.append(t_bbox)
		# case 3
		for d_index in d_rest:
			d_bbox = det_frame_index_next[d_index]
			d_and_t_bboxes.append(d_bbox)
				
		# print (d_and_t_bboxes)
		# print ret[frame_index_next_abs]

		# replace the old detection results
		# frames_ret[frame_index_next_abs] = d_and_t_bboxes
		ret_current_class[frame_index_next_abs] = d_and_t_bboxes
		# print ret_current_class[frame_index_next_abs]

		## output thre results
		for item in d_and_t_bboxes:
			outbbox = "{} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {}\n".format(item.fid, item.class_index, item.score, item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3], item.flag)
			output.write(outbbox)
			
	output.close()	
