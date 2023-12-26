import os
import argparse
from filelock import FileLock
import nltk
import re
import glob
import pandas as pd
from bert_score import BERTScorer
from rouge_score import rouge_scorer

with FileLock(".lock") as lock:
  nltk.download("punkt", quiet=True)

def save_score_df(df, file_name, args):
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
  
  pd.concat([means, medians, sds, maxs, mins], axis=1).to_csv(os.path.join(args.data_root, file_name), encoding="utf-8", index=True)

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=str, help="")
  parser.add_argument("--modelA", type=str, help="")
  parser.add_argument("--modelB", type=str, help="")
  parser.add_argument("--type", type=str, help="")
  
  args, unknown = parser.parse_known_args()
  return args, unknown

def main():
  args, unknown = read_args()

  bert = BERTScorer(lang="en-sci", rescale_with_baseline=True, model_type="allenai/scibert_scivocab_uncased")
  rouge = rouge_scorer.RougeScorer(rouge_types=["rouge1", "rouge2", "rougeLsum"], use_stemmer=True, split_summaries=True)
  res = []

  def score(hyp, ref, id, section_id=None):
    bp, br, bf1 = bert.score(cands=[hyp], refs=[ref])

    if section_id:
      row = {'article_id': id, "section_id": section_id, "BertP": bp.item(), "BertR": br.item(), "BertF1": bf1.item()}  
    else:
      row = {'article_id': id, "BertP": bp.item(), "BertR": br.item(), "BertF1": bf1.item()}
    rscores = rouge.score(target=ref, prediction=hyp)
    for key in rscores.keys():
      row[key+'P'] = rscores[key][0] * 100
      row[key+'R'] = rscores[key][1] * 100
      row[key+'F1'] = rscores[key][2] * 100

    return row

  if args.type in ("full", "model"):
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
        res.append(score(hyp_text, ref_text, id))

    else:
      for hypA in glob.glob(os.path.join(args.modelA, "hyp/hyp_*")):
        with open(hypA) as f:
          a_text = f.read()
        a_text = re.sub("\n", " ", a_text)
        id = hypA.split("/")[-1][4:-4]
        hypB = glob.glob(os.path.join(args.modelB, "hyp", "hyp_"+id+".txt"))
        if len(hypB) == 0:
          print("No hyp file for model B for article_id: ", id)
          continue
        elif len(hypB) > 1:
          print("Multiple hyp files for model B for article_id: ", id)
          continue
        with open(hypB[0]) as f:
          b_text = f.read()
        b_text = re.sub("\n", " ", b_text)
        res.append(score(a_text, b_text, id))

    df = pd.DataFrame.from_dict(res)
    save_score_df(df, "scores.csv", args)
    df.to_csv(os.path.join(args.data_root, "scores_details.csv"), encoding="utf-8", index=False)

  elif args.type=="partial":
    tdf = pd.read_csv(os.path.join(args.data_root, "partialSummaries.csv"))
    tdf = tdf.dropna()
    for index, text_row in tdf.iterrows():
      res.append(score(text_row['gen_sum'], text_row['target_sum'], text_row['article_id'], text_row["section_id"]))
    
    df = pd.DataFrame.from_dict(res)
    select_sections = ["i", "m", "r", "c"]
    save_score_df(df.loc[:, df.columns!="section_id"], "partial_unfiltered_scores.csv", args)
    save_score_df(df[df['section_id'].isin(select_sections)].loc[:, df.columns!="section_id"], "partial_filtered_scores.csv", args)
    df.to_csv(os.path.join(args.data_root, "partial_unfiltered_scores_details.csv"), encoding="utf-8", index=False)
    df[df['section_id'].isin(select_sections)].to_csv(os.path.join(args.data_root, "partial_filtered_scores_details.csv"), encoding="utf-8", index=False)

if __name__ == "__main__":
  main()
