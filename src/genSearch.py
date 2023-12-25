import argparse
import itertools
from tqdm import tqdm
import csv
import torch
from datasets import load_dataset
from datasets import load_metric

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, help="")
  parser.add_argument("--model_revision", type=str, help="")
  parser.add_argument("--tokenizer_name", type=str, help="")
  parser.add_argument("--tokenizer_revision", type=str, help="")
  parser.add_argument("--dataset", type=str, help="")
  parser.add_argument("--batch_size", type=int, help="")
  parser.add_argument("--max_src_length", type=int, help="")
  parser.add_argument("--max_target_length", type=int, help="")
  parser.add_argument("--text_col", type=str, help="")
  parser.add_argument("--sum_col", type=str, help="")
  parser.add_argument("--seed", type=int, default=10, help="")

  args, unknown = parser.parse_known_args()
  return args, unknown

def main():
  args, unknown = read_args()

  random.seed(args.seed)
  torch.manual_seed(args.seed)

  dataset = load_dataset(args.dataset, split="validation")
  params = {'batch_size': args.batch_size, 'shuffle': False}
  test_loader = torch.utils.data.DataLoader(dataset, **params)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.tokenizer_revision)
  model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, revision=args.model_revision)
  model.to(device)
  metric = load_metric("rouge")

  beam_size = [1, 2, 3]
  length_penalty = [0.8, 1, 1.2]
  no_repeat_ngram_size = [0, 3, 4]  
  res = []

  for beam, length, ngrams in itertools.product(beam_size, length_penalty, no_repeat_ngram_size):
    gen_sums = []
    #when num_beams is 1, the length penalty takes no effect. To not repeat unnecessarily, only use length penalty 1 when beam is 1
    if beam == 1 and length != 1:
      continue
    for batch in tqdm(test_loader):
      model_inputs = tokenizer(
        batch[args.text_col],
        max_length=args.max_src_length,
        truncation=True,
        padding=True,
        return_tensors='pt'
      )

      input_ids = model_inputs['input_ids'].to(device)
      if args.tokenizer_name == "allenai/led-base-16384":
        attention_mask = model_inputs["attention_mask"].to(device)
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1
        outputs = model.generate(input_ids, max_length=args.max_target_length, attention_mask=attention_mask, global_attention_mask=global_attention_mask, num_beams=beam, length_penalty=length, no_repeat_ngram_size=ngrams, early_stopping=True)
      else:
        outputs = model.generate(input_ids, max_length=args.max_target_length, num_beams=beam, length_penalty=length, no_repeat_ngram_size=ngrams, early_stopping=True)
      gen_sum = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_space=True) for g in outputs]
      gen_sums += gen_sum
    
    rscores = metric.compute(predictions=gen_sums, references=dataset[args.sum_col])
    res += [[rscores['rouge2'].mid.fmeasure, {'beam_size': beam, 'length_penalty': length, 'no_repeat_ngram_size': ngrams}]]

  with open('genSearchRes.csv', 'w+') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(res)

if __name__ == "__main__":
  main()
