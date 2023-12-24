python src/dancer_generation.py \
    --mode standard \
    --model_path google/pegasus-pubmed \
    --output_path pegasus_pubmed \
    --dataset_name scientific_papers --dataset_config_name pubmed\
    --text_column article \
    --summary_column abstract \
    --write_rouge 1 \
    --seed 100 \
    --test_batch_size 4 \
    --max_source_length 1024 --max_summary_length 256 \
    --num_beams 5