
#!/bin/bash

python test.py --fold 1 --dir_pth ./training_logs/temporal_alignment_fold_1 --fp_pkl ./training_logs/temporal_alignment_fold_1/results.pickle --net DM_MM_TCM_CM --device 0 --dir_out ./temporal_alignment_testing_results/ --save_videos --save_frames --warnings_off --summarize_off;

python test.py --fold 2 --dir_pth ./training_logs/temporal_alignment_fold_2 --fp_pkl ./training_logs/temporal_alignment_fold_2/results.pickle --net DM_MM_TCM_CM --device 0 --dir_out ./temporal_alignment_testing_results/ --save_videos --save_frames --warnings_off --summarize_off;

python test.py --fold 3 --dir_pth ./training_logs/temporal_alignment_fold_3 --fp_pkl ./training_logs/temporal_alignment_fold_3/results.pickle --net DM_MM_TCM_CM --device 0 --dir_out ./temporal_alignment_testing_results/ --save_videos --save_frames --warnings_off --summarize_off;

python test.py --fold 4 --dir_pth ./training_logs/temporal_alignment_fold_4 --fp_pkl ./training_logs/temporal_alignment_fold_4/results.pickle --net DM_MM_TCM_CM --device 0 --dir_out ./temporal_alignment_testing_results/ --save_videos --save_frames --warnings_off --summarize_off;

python test.py --fold 5 --dir_pth ./training_logs/temporal_alignment_fold_5 --fp_pkl ./training_logs/temporal_alignment_fold_5/results.pickle --net DM_MM_TCM_CM --device 0 --dir_out ./temporal_alignment_testing_results/ --save_videos --save_frames --warnings_off --summarize_off;

python test.py --fold 6 --dir_pth ./training_logs/temporal_alignment_fold_6 --fp_pkl ./training_logs/temporal_alignment_fold_6/results.pickle --net DM_MM_TCM_CM --device 0 --dir_out ./temporal_alignment_testing_results/ --save_videos --save_frames --warnings_off --summarize_off;

python test.py --fold 7 --dir_pth ./training_logs/temporal_alignment_fold_7 --fp_pkl ./training_logs/temporal_alignment_fold_7/results.pickle --net DM_MM_TCM_CM --device 0 --dir_out ./temporal_alignment_testing_results/ --save_videos --save_frames --warnings_off --summarize_off;

python test.py --fold 8 --dir_pth ./training_logs/temporal_alignment_fold_8 --fp_pkl ./training_logs/temporal_alignment_fold_8/results.pickle --net DM_MM_TCM_CM --device 0 --dir_out ./temporal_alignment_testing_results/ --save_videos --save_frames --warnings_off --summarize_off;

python test.py --fold 9 --dir_pth ./training_logs/temporal_alignment_fold_9 --fp_pkl ./training_logs/temporal_alignment_fold_9/results.pickle --net DM_MM_TCM_CM --device 0 --dir_out ./temporal_alignment_testing_results/ --save_videos --save_frames --warnings_off;
