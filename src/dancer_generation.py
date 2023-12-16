import os
import random
import argparse
from tqdm import tqdm

import torch

import scoring
import loaders

#HA: want to save partial summaries from pandas dataframe to csv
import pandas as pd


def generate_summaries(test_loader, args, device):
    """Run summary generation for a given DataLoader"""
    model, tokenizer = loaders.load_model(args, device=device)
    
    gen_sums = []
    target_sums = []
    article_ids = []
    section_ids = []
    abstracts = []
    for i, batch in enumerate(tqdm(test_loader)):
        model_inputs = tokenizer(
            batch[args.text_column],
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            return_tensors='pt')

        #HA: for LED set global attention on first token, as suggested by Beltagy et al. (2020) https://arxiv.org/abs/2004.05150
        if args.model_path == "Hendrik-a/LED-base-16384-arXiv":
            model_inputs["global_attention_mask"] = len(model_inputs["input_ids"]) * [[0 for _ in range(len(model_inputs["input_ids"][0]))]]
            model_inputs["global_attention_mask"][0][0] = 1
        
        input_ids = model_inputs['input_ids'].to(device)
        sent_outputs = model.generate(
            input_ids,
            num_beams=args.num_beams,
            early_stopping=True,
            #HA: dictionary and output_scores not used anyway, so I removed it. Should increase performance a bit, since no dict has to be returned
            #return_dict_in_generate=True,
            #output_scores=True
        )  # only one beam should be equivalent to greedy,
        
        gen_sum = [
            tokenizer.decode(
                g, skip_special_tokens=True,
                #HA: originally set to False, but I set it to True. It removes redundant white spaces
                #HA: originally sent_outputs["sequences"] because it returend a dict, but I changed it to only returning the tensors, so it had to be changed to sent_outputs
                clean_up_tokenization_spaces=False) for g in sent_outputs]

        gen_sums += gen_sum
        target_sums += batch[args.summary_column]

        #HA: in the original code, there was no if-else statement. Only the section commented out below. Reasoning in comment below
        if args.mode == "dancer":
            try:
                article_ids += batch["article_id"]
                section_ids += batch["section_id"]
                abstracts += batch["abstract"]
            except KeyError:
                article_ids += [i]
        else:
            article_ids += batch["article_id"]

        #HA: I want to also add the article_ids for the non-DANCER dataset, where there are no section_ids or abstracts. Section above already implements this, so the following should be commented out
        """
        try:
            article_ids += batch["article_id"]
            section_ids += batch["section_id"]
            abstracts += batch["abstract"]
        except KeyError:
            article_ids += [i]
            pass
        """
    return gen_sums, target_sums, article_ids, section_ids, abstracts


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="")
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--output_path", type=str, help="")
    parser.add_argument("--dataset_name", type=str, help="")
    parser.add_argument("--dataset_config_name", type=str, help="")
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--text_column", type=str, help="")
    parser.add_argument("--summary_column", type=str, help="")

    parser.add_argument("--tokenizer_name", type=str, help="")
    parser.add_argument("--max_source_length", type=int, default=512, help="")
    parser.add_argument("--max_summary_length", type=int, default=128, help="")
    parser.add_argument("--max_test_samples", type=int, help="")
    parser.add_argument("--write_rouge", type=int, default=0, help="")
    parser.add_argument("--seed", type=int, default=10, help="")
    parser.add_argument("--test_batch_size", type=int, default=2, help="")
    parser.add_argument("--num_beams", type=int, default=3, help="")

    #HA: added revisions to specify commits
    parser.add_argument("--tokenizer_revision", type=st, help="")
    parser.add_argument("--model_revision", type=st, help="")

    args, unknown = parser.parse_known_args()

    return args, unknown


def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    args, unknown = read_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    write_rouge = bool(args.write_rouge)

    select_sections = ["i", "m", "r", "c"]
    print(f"Mode: {args.mode}")
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    out_path = os.path.join(args.output_path, "generations")
    test_loader = loaders.init_loader(args)
    gen_sums, target_sums, article_ids, section_ids, abstracts = generate_summaries(test_loader, args, device=device)
    
    print("Scoring generated summaries")
    if args.mode == "dancer":        
        
        metrics = scoring.score_dancer(
            gen_sums=gen_sums,
            target_sums=abstracts,
            article_ids=article_ids,
            section_ids=section_ids,
            #HA: below the summaries are saved without filtering on section types. To distinguish between the folders, the outpath is adjusted to use "filtered" subdirectory
            out_path=os.path.join(out_path, "filtered"),
            select_sections=select_sections,
            write_gens=write_rouge)
        #HA: I want to write the files without filtering on section type as well, so another score_dancer call was added
        metrics = scoring.score_dancer(
            gen_sums=gen_sums,
            target_sums=abstracts,
            article_ids=article_ids,
            section_ids=section_ids,
            out_path=os.path.join(out_path, "unfiltered"),
            write_gens=write_rouge)

        #HA: Added the following to save the partial generated summaries with the partial target summaries to calculate scores.
        df = pd.DataFrame(
            list(zip(article_ids, section_ids, target_sums, gen_sums)),
            columns=["article_id", "section_id", "target_sum", "gen_sum"])
        df.to_csv(os.path.join(out_path, "partialSummaries.csv"), encoding="utf-8", index=False)
        
    else:
        metrics = scoring.score_standard(
            gen_sums=gen_sums,
            target_sums=target_sums,
            article_ids=article_ids,
            out_path=out_path,
            write_gens=write_rouge)

    #HA: generation of the summary files and the scoring are seperated by me. For scoring another script by me is used, which also computes the BERTScore. Therefore, teh following is commented out
    """
    if write_rouge:
        scores_dict = scoring.score_outputs(out_path)
        scoring.rouge_log(scores_dict, out_path)
    else:
        print(metrics)
    """


if __name__ == "__main__":
    main()
