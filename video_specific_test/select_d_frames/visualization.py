import os,sys
import cv2
from collections import defaultdict
from easydict import EasyDict

from vdetlib.utils.protocol import vid_proto_from_dir, frame_path_at
from vdetlib.tools.imagenet_annotation_processor import get_anno

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
	



if __name__ == '__main__':
	vid_name = sys.argv[2]
	
	print (vid_name)
	## load vid tracks
	video_foler = VID_ROOT + vid_name
	vid = vid_proto_from_dir(video_foler, vid_name)
	print (len(vid['frames']))
	
	anno_folder = ANNO_ROOT + vid_name
	annot = get_anno(anno_folder)
	print (annot.keys())
	print (annot['video'])
	print (len(annot['annotations']))
	# print (annot['annotations'][0].keys())
	# print (len(annot['annotations'][0]['track']))
	# print (annot['annotations'][0]['track'][0])
	# print (annot['annotations'][1].keys())
	# print (len(annot['annotations'][1]['track']))
	# print (annot['annotations'][1]['track'][0])

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
			fid_to_path[int(line[1])] = os.path.join(VID_ROOT, 'val', line[0] + '.JPEG')
			videos[os.path.dirname(line[0])].append(int(line[1]))

	# sort frames inside each video
	for k in videos:
		videos[k].sort()
	
	# find fids that are needed
	fids = videos[vid_name]
	print (fids)
	
	# for v2 selection
	fid_min = fids[0]
	fid_max = fids[-1]
	
	ret = defaultdict(list)
	with open(det_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split()
			# v2: use order infor
			# print (fids)
			if int(line[0]) >= fid_min and int(line[0]) <= fid_max:
				item = {
					'fid': int(line[0]),
					'class_index': int(line[1]),
					'score': float(line[2]),
					'bbox': map(float, line[3:])
				}
				item = EasyDict(item)
				ret[item.fid].append(item)
				# print (len(fids))
			if int(line[0]) > fid_max:
				break

			# v1: use all fids when found
			# if int(line[0]) in fids:
			# 	item = {
			# 		'fid': int(line[0]),
			# 		'class_index': int(line[1]),
			# 		'score': float(line[2]),
			# 		'bbox': map(float, line[3:])
			# 	}
			# 	item = EasyDict(item)
			# 	ret[item.fid].append(item)
			# 	# print (len(fids))
			# 	fids.remove(int(line[0]))
			# 	if len(fids) == 0:
			# 		print ("get all fids")
			# 		break
	print (len(ret))
	
	## match the gt with detection for each video
	## show the gt
	
	# remove the current images
	os.system("rm saveImgs/*")

	for frame_index, frame in enumerate(vid['frames']):
		# print (frame)
		# print (vid['root_path'])
		imgpath = frame_path_at(vid, frame['frame'])
		imgbasename = os.path.basename(imgpath)
		imgsavepath = os.path.join('saveImgs/', imgbasename)
		img = cv2.imread(imgpath)

		# load gt boundbox and annotation 
		# print (annot['annotations'])
	        boxes = [track_box_at_frame(tracklet, frame['frame']) for tracklet in [anno['track'] for anno in annot['annotations']]]
	        classes = [track_class_at_frame(tracklet, frame['frame']) for tracklet in [anno['track'] for anno in annot['annotations']]]
		# print (boxes)
		# print (classes)

		# get ground truth
		# print (fids[frame_index])
		preds = ret[fids[frame_index]]

		# print (preds)

		## find the match one, started from objects
		for pred in preds:
			ovmax = -1
			kmax = -1
			count = 0
			keytype = -1
			pred_class_ = CLASS_NAMES[pred.class_index]

			# get max IoU and then judge

			for bbox_, class_ in zip(boxes, classes):
				if bbox_ != None and class_ != None:
					# draw the GT with yellow
					x1, y1, x2, y2 = map(int, bbox_)
					
			 		cv2.rectangle(img, (x1,y1), (x2,y2), COLORS.blue, 3)
					cv2.rectangle(img, (x1,y1), (np.min((x2, x1 + 200)), y1 - 30), COLORS.blue, cv2.FILLED) 
					draw_gt_class = 'GT: ' + class_
			 		cv2.putText(img, draw_gt_class, (x1+4,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FontColor, 2)
					# 
					gt_thr = get_gt_thres(bbox_)
	
					# print pred, bbox_
					# calculate the IoU of two box
					IoU = cal_IoU(bbox_, pred.bbox)
					# print IoU, gt_thr
					print IoU, frame['frame']
					# print pred.class_index, CLASS_NAMES[pred.class_index], class_
					
					# case 1 : correct detection, 0.5 <= IoU       + correct class label
					# case 2 : Error 1,             0 <  IoU < 0.5 + correct class label
					# case 3 : Error 2,           0.5 <= IoU       + wrong class label
					# case 4 : Error 3,             0 <  IoU < 0.5 + wrong class label
					# if keytype == 1:
					# 	break
					if IoU > ovmax:
						ovmax = IoU
						gt_select = class_

			# get type
			IoU = ovmax
			class_ = gt_select
			if IoU >= gt_thr and class_ == pred_class_:
				# case 1
				if IoU >= ovmax:
					ovmax = IoU
					keytype = 1
			elif IoU > 0.0 and IoU < gt_thr and class_ == pred_class_ and keytype != 1:
				# case 2
				keytype = 2
			elif IoU >= gt_thr and class_ != pred_class_ and keytype != 1:
				# case 3
				keytype = 3
			elif IoU > 0.0 and IoU < gt_thr and class_ != pred_class_ and keytype != 1:
				# case 3
				keytype = 4
			elif keytype != 1:
				keytype = 5
			else:
                                keytype = keytype
				# print("non from case 1 to 4")
						# keytype = 1 

			x1, y1, x2, y2 = map(int, pred.bbox)
			pred_class_score = '{}{} {:.2f}'.format('DET:', pred_class_, pred.score)
			if keytype == 1:
						cv2.rectangle(img, (x1,y1), (x2,y2), COLORS.green, 3)
						cv2.rectangle(img, (x1,y2), (np.min((x2, x1 + 200)), y2 - 30), COLORS.green, cv2.FILLED) 
				 		cv2.putText(img, pred_class_score, (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FontColor, 2)
			elif keytype == 2:
						cv2.rectangle(img, (x1,y1), (x2,y2), COLORS.yellow, 3)
						cv2.rectangle(img, (x1,y2), (np.min((x2, x1 + 200)), y2 - 30), COLORS.yellow, cv2.FILLED) 
				 		cv2.putText(img, pred_class_score, (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FontColor, 2)
			elif keytype == 3:
						cv2.rectangle(img, (x1,y1), (x2,y2), COLORS.red, 3)
						cv2.rectangle(img, (x1,y2), (np.min((x2, x1 + 200)), y2 + 30), COLORS.red, cv2.FILLED) 
				 		cv2.putText(img, pred_class_score, (x1 + 4, y2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FontColor, 2)
			elif keytype == 4:
						cv2.rectangle(img, (x1,y1), (x2,y2), COLORS.gray, 3)
						cv2.rectangle(img, (x1,y2), (np.min((x2, x1 + 200)), y2 - 30), COLORS.gray, cv2.FILLED) 
				 		cv2.putText(img, pred_class_score, (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FontColor, 2)
			elif keytype == -1:
				# miss or no gt in the frame
				if pred.score > 0.01:
						cv2.rectangle(img, (x1,y1), (x2,y2), COLORS.black, 3)
						cv2.rectangle(img, (x1,y2), (np.min((x2, x1 + 200)), y2 - 30), COLORS.black, cv2.FILLED) 
				 		cv2.putText(img, pred_class_score, (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
			else:
				assert(keytype==5)
				
		cv2.imwrite(imgsavepath, img)
						
		# cv2.imshow('abc', img)
		# cv2.waitKey()

		# exit()



	
	
	

	

	
