## get results of current video first

python visualization.py det_remove_lowschores-0.1 ILSVRC2015_val_00159001


## selected frame id for detection, manual 
write into a file, e.g. manual_selected_key_frame_index


# 
python filter_frame.py det_remove_lowschores-0.1 ILSVRC2015_val_00159001 manual_selected_key_frame_index video_result
