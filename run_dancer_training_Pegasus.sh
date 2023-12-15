python src/run_summarization.py \
    --model_name_or_path google/pegasus-large --tokenizer_name google/pegasus-large \
    --do_train \
    --do_eval \
    --task summarization \
    --dataset_name Hendrik-a/dancer-data \
    --text_column document \
    --summary_column summary \
    --seed 100 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --predict_with_generate \
    --learning_rate 1e-4 \
    --adafactor \
    --max_source_length 1024 --max_target_length 128 --val_max_target_length 128 --pad_to_max_length True \
    --num_beams 1 \
    --tf32 True \
    --lr_scheduler_type constant \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps 300 \
    --eval_steps 300 \
    --logging_steps 100 \
    --max_steps 100000000000000 \
    --training_duration 1
    
