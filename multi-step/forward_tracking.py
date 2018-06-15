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


def data_reader(input_queue):
    '''
    this function only reads the image from the queue
    '''
    search_img = tf.read_file(input_queue[1])
    target_img = tf.read_file(input_queue[0])
    search_tensor = tf.to_float(tf.image.decode_jpeg(search_img, channels = 3))
    target_tensor = tf.to_float(tf.image.decode_jpeg(target_img, channels = 3))

    # no need to resize
    search_tensor = tf.image.resize_images(search_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    target_tensor = tf.image.resize_images(target_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    return [search_tensor, target_tensor]


def next_batch(input_queue):
    min_queue_examples = 128
    num_threads = 4
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
    # 
    test_list = sys.argv[1]
    bboxes_current = []
    bboxes_next = []
    input_dir = 'tmp_images/'
    with open(test_list, 'r') as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip().split()[0]
        bboxes_current.append(os.path.join(input_dir, 'current', line))
        bboxes_next.append(os.path.join(input_dir, 'next', line))


    num_bboxes = len(bboxes_current)
     
    bboxes_current_tensors = tf.convert_to_tensor(bboxes_current, dtype=tf.string)
    bboxes_next_tensors = tf.convert_to_tensor(bboxes_next, dtype=tf.string)

    input_queue = tf.train.slice_input_producer([bboxes_current_tensors, bboxes_next_tensors], shuffle=False)
    batch_queue = next_batch(input_queue)

    ## load model 
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
        # next_bboxes = []
        output = open(sys.argv[2], 'w')
        total_batch = int(np.ceil(1.0*num_bboxes/BATCH_SIZE))
        for i in range(0, total_batch):
            print (i)
            start_time = time.time()
            cur_batch = sess.run(batch_queue)
            [fc4] = sess.run([tracknet.fc4],feed_dict={tracknet.image:cur_batch[0],
                    tracknet.target:cur_batch[1]})
            for batch_item in range(fc4.shape[0]):
              x1 = (227* fc4[batch_item][0]/10)
              y1 = (227* fc4[batch_item][1]/10)
              x2 = (227* fc4[batch_item][2]/10)
              y2 = (227* fc4[batch_item][3]/10)
              outbbox = "{} {} {} {}\n".format(x1, y1, x2, y2)
              output.write(outbbox)

              # next_bboxes.append([x1,y1,x2,y2])
            print ('batch %d/%d, time elapsed: %.3fs.'%(i, total_batch, time.time()-start_time))
        output.close()
        # print (len(next_bboxes))

    except KeyboardInterrupt:
        print("get keyboard interrupt")

        
        # print ('fid: %d, net forward test: time elapsed: %.3fs.'%(frame_index ,time.time()-start_time))
        ## save the result for the next frame
        # targetbox = np.float32([[56.75, 56.75], [56.75,170.25], [170.25, 170.25]])

        # for next_bbox, current_bbox in zip(next_bboxes,ret[frame_index]):
        #   # print (next_bbox, current_bbox)
        #   bbox = current_bbox.bbox
        #   pts = np.float32([ [bbox[0],bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]] ])
        #   M_inv = cv2.getAffineTransform(targetbox , pts)
        #   pts_new = np.float32( [[next_bbox[0], next_bbox[1], 1], [next_bbox[2], next_bbox[3], 1]])
        #   # warp back the bboxes
        #   original_bbox = np.matmul(M_inv, pts_new.T)
        #   tracked_bbox = [original_bbox[0][0], original_bbox[1][0], original_bbox[0][1], original_bbox[1][1]]
        #   item = {
        #           'fid': current_bbox.fid+1,
        #           'class_index': current_bbox.class_index,
        #           'score': current_bbox.score,
        #           'bbox': map(float, tracked_bbox)
        #         }
        #   # print (item)
        #   item = EasyDict(item)
        #   outbbox = "{} {} {} {} {} {} {}\n".format(item.fid, item.class_index, item.score, item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3])
        #   output.write(outbbox)
        # print ('fid: %d with %d bboxes in video %s, time elapsed: %.3fs.'%(frame_index , len(next_bboxes), vid_current, time.time()-start_time))
        # # clean memory
    # output.close()
