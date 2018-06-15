## save the images first then do the forward

1. save_patches.py
2. ls tmp_images/next/* > tmpdata/test.list
3. forward_tracking.py tmpdata/test.list tmpdata/test.output
4. paste test.list test.output > test.list.output
5. python bbox_map_back.py tmpdata/det_remove_lowschores-0.1 tmpdata/test.list.output tmpdata/det_remove_lowschores-0.1_tracking

