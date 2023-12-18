python src/dancer_generation.py \
    --mode standard \
    --model_path Hendrik-a/led-base-16384-arxiv \
    --model_revision c92038e4ca52ccf3ef0c2bce4d9711c01c9febbd \
    --tokenizer_name google/pegasus-x-base \
    --tokenizer_revision cd8a69a4c88a469312423199b8dcd331f6af8b08 \
    --output_path pX \
    --dataset_name Hendrik-a/arxiv \
    --text_column article_text \
    --summary_column abstract_text \
    --write_rouge 1 \
    --seed 100 \
    --test_batch_size 8 \
    --max_source_length 16384 --max_summary_length 256 \
    --num_beams 1 \
    --no_repeat_ngram_size 0 \
    --length_penalty 1.0 \
    --split validation
    
