python src/dancer_generation.py \
  --model_path Hendrik-a/led-base-16384-arxiv2 \
  --model_revision 7c9652e75b33689423bde82ae35eb10d013fccb0 \
  --tokenizer_name allenai/led-base-16384 \
  --tokenizer_revision 38335783885b338d93791936c54bb4be46bebed9 \
  --max_source_length 16834 \
  --max_target_length 256 \
  --output_path genLED \
  --dataset_name Hendrik-a/arxiv \
  --text_column article_text \
  --summary_column abstract_text \
  --write_rouge 1 \
  --seed 100 \
  --test_batch_size 8 \
  --num_beams 3 \
  --length_penalty 1.0 \
  --no_repeat_ngram_size 4 \
  --max_test_samples 16