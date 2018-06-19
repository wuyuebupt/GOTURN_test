# train file

import sys
import logging
import time
import tensorflow as tf
import os
import goturn_net
import cv2
import numpy as np


from collections import defaultdict
from easydict import EasyDict

# tensorflow config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# parameters
BATCH_SIZE = 1
# BATCH_SIZE = 10
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



def data_reader(input_queue):
    '''
    this function only reads the image from the queue
    '''
    # search_img = tf.read_file(input_queue[0])
    # target_img = tf.read_file(input_queue[1])
    # search_tensor = tf.to_float(tf.image.decode_jpeg(search_img, channels = 3))
    # target_tensor = tf.to_float(tf.image.decode_jpeg(target_img, channels = 3))
    search_tensor = input_queue[0]
    target_tensor = input_queue[1]

    search_tensor = tf.image.resize_images(search_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    target_tensor = tf.image.resize_images(target_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    return [search_tensor, target_tensor]


def next_batch(input_queue):
    min_queue_examples = 128
    num_threads = 8
    [search_tensor, target_tensor] = data_reader(input_queue)
    [search_batch, target_batch] = tf.train.batch(
        [search_tensor, target_tensor],
        batch_size=BATCH_SIZE,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads+2)*BATCH_SIZE)
    return [search_batch, target_batch]

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
    ## finish get bboxes
    assert (len(ret.keys()) == 1)

    ## get two images
    key = 1 # the first image
    img_current_path = fid_to_path[key]
    img_next_path = fid_to_path[key+1]
    
    # generate two pairs of image patches for tracking 
    # target is in the middle
    # search is for the tracking 
    img_current = cv2.imread(img_current_path)
    img_next = cv2.imread(img_next_path)

    ## finish getting two images


    ## generate testing patches
    # put the object in the center of the input
    # cv2.imshow('abc',img_t)
    # cv2.waitKey()
    # cv2.imshow('def',img_s)
    # cv2.waitKey()

    targetbox = np.float32([[56.75, 56.75], [56.75,170.25], [170.25, 170.25]])

    test_pairs = []
    for count, item in enumerate(ret[key]):
        bbox = item.bbox 
        pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
        # get warpped image
        M = cv2.getAffineTransform(pts ,targetbox)
        bbox_current = cv2.warpAffine(img_current, M, (WIDTH, HEIGHT))
        bbox_next    = cv2.warpAffine(img_next, M, (WIDTH, HEIGHT))
        ## test: map the points
        # pts_new = np.float32( [[bbox[0], bbox[1], 1], [bbox[2], bbox[3], 1]])
        # print (pts_new.shape)
        # print (M.shape)
        # print (np.matmul(M, pts_new.T))
        # print (pts_new[0])
        # target_points = np.matmul(M, pts_new.T)
        # print (target_points.shape)
        # M_inv = cv2.getAffineTransform(targetbox , pts)
        # pts_new = np.float32([[56.75, 56.75, 1], [170.25, 170.25, 1]])
        # original_points = np.matmul(M_inv, pts_new.T)
        # print (original_points, pts)
        # exit()

        # test pair : patch, patch, patch position in original images
        test_pairs.append([bbox_current, bbox_next])
        # cv2.imshow('c',bbox_target)
        # cv2.waitKey()
        # cv2.imshow('d',bbox_search)
        # cv2.waitKey()
    print len(test_pairs)
    
    ## generate test batch and do the forward

    train_target = [x[0] for x in test_pairs]
    train_search = [x[1] for x in test_pairs]

    train_target = np.asarray(train_target, np.float64)
    train_search = np.asarray(train_search, np.float64)

    target_tensors = tf.convert_to_tensor(train_target, dtype=tf.float64)
    search_tensors = tf.convert_to_tensor(train_search, dtype=tf.float64)
    target_tensors_resize = tf.image.resize_images(target_tensors,[HEIGHT,WIDTH],method=tf.image.ResizeMethod.BILINEAR)
    search_tensors_resize = tf.image.resize_images(search_tensors,[HEIGHT,WIDTH],method=tf.image.ResizeMethod.BILINEAR)

    tracknet = goturn_net.TRACKNET(BATCH_SIZE, train = False)
    tracknet.build()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)

    coord = tf.train.Coordinator()
    # start the threads
    tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt_dir = "./checkpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
    try:
        # for i in range(0, int(len(train_box)/BATCH_SIZE)):

        next_bboxes = []
        # cur_batch = sess.run(batch_queue)

        start_time = time.time()

        target_tensors = sess.run(target_tensors)
        search_tensors = sess.run(search_tensors)
        [fc4] = sess.run([tracknet.fc4],feed_dict={tracknet.image:search_tensors,
                tracknet.target:target_tensors})

        # target_tensors_resize = sess.run(target_tensors_resize)
        # search_tensors_resize = sess.run(search_tensors_resize)
        #[fc4] = sess.run([tracknet.fc4],feed_dict={tracknet.image:search_tensors_resize,
        #        tracknet.target:target_tensors_resize})
        print (fc4)
        # print (cur_batch[0])
        # logging.info('batch box: %s' %(fc4))
        # logging.info('gt batch box: %s' %(cur_batch[2]))
        # logging.info('batch loss = %f'%(batch_loss))
        print ('test: time elapsed: %.3fs.'%(time.time()-start_time))
        # print (cur_batch[0].shape)
        # img = cur_batch[0].reshape(227,227,3)/255
        # img2 = cur_batch[1].reshape(227,227,3)/255
        
        # img = np.rollaxis(img, 1, 0)  
        # print (img.shape)
        # print (fc4)
        for i in range(len(test_pairs)):
          x1 = int(227* fc4[0][0]/10)
          y1 = int(227* fc4[0][1]/10)
          x2 = int(227* fc4[0][2]/10)
          y2 = int(227* fc4[0][3]/10)
          # x1 = (227* fc4[0][0]/10)
          # y1 = (227* fc4[0][1]/10)
          # x2 = (227* fc4[0][2]/10)
          # y2 = (227* fc4[0][3]/10)
          next_bboxes.append([x1,y1,x2,y2])
          # cv2.rectangle(img, (x1,y1), (x2,y2), (255,120,120), 1)
          # cv2.imshow('abc', img)
          # cv2.waitKey()
          # exit()
    except KeyboardInterrupt:
        print("get keyboard interrupt")

    print (len(next_bboxes))

    # visualize the tracking results in the next frame
    targetbox = np.float32([[56.75, 56.75], [56.75,170.25], [170.25, 170.25]])
    img = img_next
    for next_bbox, current_bbox in zip(next_bboxes,ret[key]):
      # print (next_bbox, current_bbox)
      print (next_bbox, current_bbox)
      bbox = current_bbox.bbox
      pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
      M_inv = cv2.getAffineTransform(targetbox , pts)
      # print (M_inv)
      # next_bbox
      pts_new = np.float32( [[next_bbox[0], next_bbox[1], 1], [next_bbox[2], next_bbox[3], 1]])
      original_bbox = np.matmul(M_inv, pts_new.T)
      # print (original_bbox)
      tracked_bbox = [original_bbox[0][0], original_bbox[1][0], original_bbox[0][1], original_bbox[1][1]]
      item = {
                'fid': current_bbox.fid+1,
                'class_index': current_bbox.class_index,
                'score': current_bbox.class_index,
                'bbox': map(float, tracked_bbox)
              }
      print (item)
      x1 = int(original_bbox[0][0])
      y1 = int(original_bbox[1][0])
      x2 = int(original_bbox[0][1])
      y2 = int(original_bbox[1][1])
      cv2.rectangle(img, (x1,y1), (x2,y2), (255,120,120), 1)
      cv2.imshow('abc', img)
      cv2.waitKey()
      # original_points = np.matmul(M_inv, target_points)
    # 




