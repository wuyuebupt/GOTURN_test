# train file
# video forward tracking using detection results from previous frames

import sys
import logging
import time
import tensorflow as tf
import os
import goturn_net
import cv2
import numpy as np
import gc

from collections import defaultdict
from easydict import EasyDict

# tensorflow config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

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
    num_threads = 2
    [search_tensor, target_tensor] = data_reader(input_queue)
    [search_batch, target_batch] = tf.train.batch(
        [search_tensor, target_tensor],
        batch_size=BATCH_SIZE,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads+2)*BATCH_SIZE)
    # print ('next_batch done')
    return [search_batch, target_batch]

if __name__ == "__main__":
    # load det results in one frame
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
    sorted_det_keys = sorted(ret.keys())
    print (sorted_det_keys)

    ## open the output file
    output = open(sys.argv[2], 'w')

    ## load model 
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

    # prepare data in one frame
    forward_step = 1

    assert (len(sorted_det_keys) >= 2)
    # the first and the last 
    while True:
      frame_index = sorted_det_keys[0]
      frame_index_next = frame_index - 1

      # first frame 
      if frame_index_next <= 0:
        print ("the last frame")
        break

      # loop for forward tracking 
      while True:
        ## check if two images are from the same video
        img_current_path = fid_to_path[frame_index]
        img_next_path = fid_to_path[frame_index_next]
        vid_current = img_current_path.split('/')[-2]
        vid_next    = img_next_path.split('/')[-2]
        if vid_current == vid_next:
          # track two frames 
          # input: two frames, initial bboxes, one model
          # return: predicted bboxes
          start_time = time.time()
  
          img_current = cv2.imread(img_current_path)
          img_next = cv2.imread(img_next_path)
          targetbox = np.float32([[56.75, 56.75], [56.75,170.25], [170.25, 170.25]])
  
          patches_current = []
          patches_next = []
          for item in ret[frame_index]:
              bbox = item.bbox 
              pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
              # get warpped image
              M = cv2.getAffineTransform(pts ,targetbox)
              bbox_current = cv2.warpAffine(img_current, M, (WIDTH, HEIGHT))
              bbox_next    = cv2.warpAffine(img_next, M, (WIDTH, HEIGHT))
              cv2.imshow('abc',bbox_current)
              cv2.imshow('def',bbox_next)
              cv2.waitKey()
              patches_current.append(bbox_current)
              patches_next.append(bbox_next)
          num_bboxes = len(patches_current)
          print (num_bboxes)
          if num_bboxes == 0: 
            break
  
          ## generate test batch and do the forward
          patches_current = np.asarray(patches_current, np.float64)
          patches_next = np.asarray(patches_next, np.float64)
          
          patches_current_tensors = sess.run(tf.convert_to_tensor(patches_current, dtype=tf.float64))
          patches_current_tensors = sess.run(tf.image.resize_images(patches_current_tensors,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR))
          patches_next_tensors = sess.run(tf.convert_to_tensor(patches_next, dtype=tf.float64))
          patches_next_tensors = sess.run(tf.image.resize_images(patches_next_tensors,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR))
          print (patches_current_tensors.shape)
          fc4 = sess.run(tracknet.fc4, feed_dict={tracknet.image:patches_next_tensors, tracknet.target:patches_current_tensors})
          print (fc4)

          next_bboxes = []
          for i in range(num_bboxes):
              x1 = (227* fc4[i][0]/10)
              y1 = (227* fc4[i][1]/10)
              x2 = (227* fc4[i][2]/10)
              y2 = (227* fc4[i][3]/10)
              next_bboxes.append([x1,y1,x2,y2])
          ## save the result for the next frame

          ret[frame_index_next] = []
          for next_bbox, current_bbox in zip(next_bboxes,ret[frame_index]):
            bbox = current_bbox.bbox
            pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
            M_inv = cv2.getAffineTransform(targetbox , pts)
            pts_new = np.float32( [[next_bbox[0], next_bbox[1], 1], [next_bbox[2], next_bbox[3], 1]])
            # warp back the bboxes
            original_bbox = np.matmul(M_inv, pts_new.T)
            tracked_bbox = [original_bbox[0][0], original_bbox[1][0], original_bbox[0][1], original_bbox[1][1]]
            item = {
                    'fid': current_bbox.fid - 1 ,
                    'class_index': current_bbox.class_index,
                    'score': current_bbox.score,
                    'bbox': map(float, tracked_bbox)
                  }
            # print (item)
            item = EasyDict(item)
            ret[frame_index_next].append(item)
            outbbox = "{} {} {} {} {} {} {}\n".format(item.fid, item.class_index, item.score, item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3])
            output.write(outbbox)
            print (outbbox)
            print ('fid: %d with %d bboxes in video %s, time elapsed: %.3fs.'%(frame_index , len(next_bboxes), vid_current, time.time()-start_time))
          # clean memory

          # update to next frames
          frame_index = frame_index_next
          frame_index_next = frame_index - 1
          if frame_index_next <= 0:
            print ("the first frame")
            break
        else:
          # went into another video
          break
      break

   # last frame

    output.close()
    exit()
    dataset = None
    for frame_index in ret.keys():
      img_current_path = fid_to_path[frame_index]
      frame_index_next = frame_index + forward_step

      # last frame 
      if frame_index_next > length_fids:
        print ("the last frame")
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

        patches_current = []
        patches_next = []
        for item in ret[frame_index]:
            bbox = item.bbox 
            pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
            # get warpped image
            M = cv2.getAffineTransform(pts ,targetbox)
            bbox_current = cv2.warpAffine(img_current, M, (WIDTH, HEIGHT))
            bbox_next    = cv2.warpAffine(img_next, M, (WIDTH, HEIGHT))
            patches_current.append(bbox_current)
            patches_next.append(bbox_next)
        num_bboxes = len(patches_current)
        # print (num_bboxes)


        ## generate test batch and do the forward
        patches_current = np.asarray(patches_current, np.float64)
        patches_next = np.asarray(patches_next, np.float64)
        # print (patches_current.shape)
        # print (patches_next.shape)

        # target_tensors = tf.convert_to_tensor(patches_current, dtype=tf.float64)
        # search_tensors = tf.convert_to_tensor(patches_next, dtype=tf.float64)
        # input_queue = tf.train.slice_input_producer([search_tensors, target_tensors],shuffle=False)

        if dataset != None and frame_index % 100 == 0:
          del dataset
          gc.collect()
        # memory leak in tensorflo
        dataset = tf.data.Dataset.from_tensor_slices((patches_current, patches_next)) 
        print ('fid: %d '%(frame_index))
        continue
        # dataset = dataset.map(lambda patch_current, patch_next: return (tf.image.resize_images(patch_current, [HEIGHT,WIDTH], method=tf.image.ResizeMethod.BILINEAR), tf.image.resize_images(patch_next,[HEIGHT,WIDTH], method=tf.image.ResizeMethod.BILINEAR)))
        # dataset = dataset.map(lambda patch_current, patch_next: img_resize(patch_current, patch_next))
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(1)
        # print (dataset)
        iterator = dataset.make_one_shot_iterator()
        # print ('fid: %d, dataset test: time elapsed: %.3fs.'%(frame_index ,time.time()-start_time))
        # start_time = time.time()
        # next_element = iterator.get_next()
        # print (next_element)
        # exit()
        # start the threads
        # coord = tf.train.Coordinator()
        # tf.train.start_queue_runners(sess=sess, coord=coord)
        next_bboxes = []
        # assert(BATCH_SIZE==1)
        # forward all bboxes 
        for i in range(0, int(np.ceil(1.0*num_bboxes/BATCH_SIZE))):
            # cur_batch = sess.run(batch_queue])
            # start_time = time.time()

            cur_batch = sess.run([iterator.get_next()])
            # print ('fid: %d, get batch: time elapsed: %.3fs.'%(frame_index ,time.time()-start_time))
            # start_time = time.time()


            # [fc4] = sess.run([tracknet.fc4],feed_dict={tracknet.image:cur_batch[0],
            #         tracknet.target:cur_batch[1]})
            # [fc4] = []
            break
            # print (fc4.shape)
            for batch_item in range(fc4.shape[0]):
              x1 = (227* fc4[batch_item][0]/10)
              y1 = (227* fc4[batch_item][1]/10)
              x2 = (227* fc4[batch_item][2]/10)
              y2 = (227* fc4[batch_item][3]/10)
              next_bboxes.append([x1,y1,x2,y2])
            # print ('fid: %d, forward test: time elapsed: %.3fs.'%(frame_index ,time.time()-start_time))
        # print (len(next_bboxes))

        
        # print ('fid: %d, net forward test: time elapsed: %.3fs.'%(frame_index ,time.time()-start_time))
        # start_time = time.time()
        ## save the result for the next frame
        targetbox = np.float32([[56.75, 56.75], [56.75,170.25], [170.25, 170.25]])

        for next_bbox, current_bbox in zip(next_bboxes,ret[frame_index]):
          # print (next_bbox, current_bbox)
          bbox = current_bbox.bbox
          pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
          M_inv = cv2.getAffineTransform(targetbox , pts)
          pts_new = np.float32( [[next_bbox[0], next_bbox[1], 1], [next_bbox[2], next_bbox[3], 1]])
          # warp back the bboxes
          original_bbox = np.matmul(M_inv, pts_new.T)
          tracked_bbox = [original_bbox[0][0], original_bbox[1][0], original_bbox[0][1], original_bbox[1][1]]
          item = {
                  'fid': current_bbox.fid+1,
                  'class_index': current_bbox.class_index,
                  'score': current_bbox.score,
                  'bbox': map(float, tracked_bbox)
                }
          # print (item)
          item = EasyDict(item)
          outbbox = "{} {} {} {} {} {} {}\n".format(item.fid, item.class_index, item.score, item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3])
          output.write(outbbox)
        print ('fid: %d with %d bboxes in video %s, time elapsed: %.3fs.'%(frame_index , len(next_bboxes), vid_current, time.time()-start_time))
        # clean memory

    output.close()
          




