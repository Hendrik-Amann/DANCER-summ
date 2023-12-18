import os
import argparse
from filelock import FileLock
import nltk
import re
import glob
import pandas as pd
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from datasets import load_metric

with FileLock(".lock") as lock:
  nltk.download("punkt", quiet=True)

def save_score_df(df, file_name):
  means= df[df.columns.difference(['article_id'])].mean()
  means.name="mean"
  medians= df[df.columns.difference(['article_id'])].median()
  medians.name="median"
  sds= df[df.columns.difference(['article_id'])].std()
  sds.name="sd"
  maxs = df[df.columns.difference(['article_id'])].max()
  maxs.name="max"
  mins = df[df.columns.difference(['article_id'])].min()
  mins.name="min"
  
  pd.concat([means, medians, sds, maxs, mins], axis=1).to_csv(os.path.join(args.data_root, filename), encoding="utf-8", index=True)

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=str, help="")
  parser.add_argument("--type", type=str, help="")
  
  args, unknown = parser.parse_known_args()
  return args, unknown

def main():
  args, unknown = read_args()

  bert = BERTScorer(lang="en-sci", rescale_with_baseline=True, model_type="allenai/scibert_scivocab_uncased")
  rouge = rouge_scorer.RougeScorer(rouge_types=["rouge1", "rouge2", "rougeLsum"], use_stemmer=True, split_summaries=True)
  res = []

  if args.type=="full":
    for ref in glob.glob(os.path.join(args.data_root, "ref/ref_*")):
      with open(ref) as f:
        ref_text = f.read()
      ref_text = re.sub("\n", " ", ref_text)
      id = ref.split("/")[-1][4:-4]
      hyp = glob.glob(os.path.join(args.data_root, "hyp", "hyp_"+id+".txt"))
      if len(hyp) == 0:
        print("No hyp file for article_id: ", id)
        continue
      elif len(hyp) > 1:
        print("Multiple hyp files for article_id:", id)
        continue
      with open(hyp[0]) as f:
        hyp_text = f.read()
      hyp_text = re.sub("\n", " ", hyp_text)

      bp, br, bf1 = bert.score(cands=[hyp_text], refs=[ref_text])
      row = {'article_id': id, "BertP": bp.item(), "BertR": br.item(), "BertF1": bf1.item()}

      #load_metric is a wrapper around rouge. Used load_metric instead of rouge here since it offers aggregated results
      metric = load_metric("rouge")
      print("Rouge scores across generated summaries:", metric.compute(predictions=decoded_preds, references=[decoded_preds[0] for i in range(len(decoded_preds))]))
      
      rscores = rouge.score(target=ref_text, prediction=hyp_text)
      for key in rscores.keys():
        row[key+'P'] = rscores[key][0] * 100
        row[key+'R'] = rscores[key][1] * 100
        row[key+'F1'] = rscores[key][2] * 100

      res.append(row)

    df = pd.DataFrame.from_dict(res)
    save_score_df(df, "scores.csv")
    df.to_csv(os.path.join(args.data_root, "scores_details.csv"), encoding="utf-8", index=False)

  elif args.type=="partial":
    tdf = pd.read_csv(os.path.join(args.data_root, "partialSummaries.csv"))
    tdf = tdf.dropna()
    for index, text_row in tdf.iterrows():
      bp, br, bf1 = bert.score(cands=[text_row['gen_sum']], refs=[text_row['target_sum']])
      score_row = {'article_id': text_row['article_id'], "BertP": bp.item(), "BertR": br.item(), "BertF1": bf1.item()}
      rscores = rouge.score(target=text_row['target_sum'], prediction=text_row['gen_sum'])
      for key in rscores.keys():
        score_row[key+'P'] = rscores[key][0] * 100
        score_row[key+'R'] = rscores[key][1] * 100
        score_row[key+'F1'] = rscores[key][2] * 100
      res.append(score_row)
    
    df = pd.DataFrame.from_dict(res)
    select_sections = ["i", "m", "r", "c"]
    save_score_df(df, "partial_unfiltered_scores.csv")
    save_score_df(df[df['section_id'].isin(select_sections)], "partial_filtered_scores.csv")
    df.to_csv(os.path.join(args.data_root, "partial_unfiltered_scores_details.csv"), encoding="utf-8", index=False)
    df[df['section_id'].isin(select_sections)].to_csv(os.path.join(args.data_root, "partial_filtered_scores_details.csv"), encoding="utf-8", index=False)

if __name__ == "__main__":
  main()
