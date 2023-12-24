python src/genSearch.py \
  --model_name Hendrik-a/pegasus-x-base-arxiv \
  --model_revision be2f12575b4d281f7f584ea40b8bf6702e1394e6 \
  --tokenizer_name google/pegasus-x-base \
  --tokenizer_revision cd8a69a4c88a469312423199b8dcd331f6af8b08 \
  --dataset Hendrik-a/arxiv \
  --batch_size 8 \
  --max_src_length 16384 \
  --max_target_length 256
