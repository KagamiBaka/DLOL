nohup bash run_seq2seq_verbose.bash -f tree --label_smoothing 0.1 -l 5e-5 --lr_scheduler linear -b 16 --seed 42 -d 0 --is_fp16 1 --wo_constraint_decoding