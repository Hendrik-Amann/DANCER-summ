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
        input_ids = model_inputs['input_ids'].to(device)
        if args.tokenizer_name == "allenai/led-base-16384":
            attention_mask = model_inputs["attention_mask"].to(device)
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1
            sent_outputs = model.generate(
                input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=True,
                max_new_tokens=max_target_length,
            )
        else:
            sent_outputs = model.generate(
                input_ids,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=True,
                #HA: added max_new_tokens (if not set the model config controlls it)
                max_new_tokens=max_target_length,
            #HA: dictionary and output_scores not used anyway, so I removed it. Should increase performance a bit, since no dict has to be returned
            #return_dict_in_generate=True,
            #output_scores=True
            )  # only one beam should be equivalent to greedy,
        
        gen_sum = [
            tokenizer.decode(
                g, skip_special_tokens=True,
                #HA: originally set to False, but I set it to True. It removes redundant white spaces
                #HA: originally sent_outputs["sequences"] because it returend a dict, but I changed it to only returning the tensors, so it had to be changed to sent_outputs
                clean_up_tokenization_spaces=True) for g in sent_outputs]

        gen_sums += gen_sum
        target_sums += batch[args.summary_column]
        try:
            article_ids += batch["article_id"]
            section_ids += batch["section_id"]
            #HA: changed it from batch["abstract"] to batch[args.summary_column]
            abstracts += batch[args.summary_column]
        except KeyError:
            #HA: with non-DANCER dataset that does not have a section_id, there will be a keyerror. Adding i as the article_id does not make sense in that case. Because it is not needed with my datasets, the following line is commented out
            #article_ids += [i]
            pass

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
    #HA: added max_target_length, no_repeat_ngram_size and length penalty
    parser.add_argument("--max_target_length", type=float, default=3, help="")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="")
    parser.add_argument("--length_penalty", type=float, default=3, help="")

    #HA: added revisions to specify commits
    parser.add_argument("--tokenizer_revision", type=str, help="")
    parser.add_argument("--model_revision", type=str, help="")

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

    #HA: generation of the summary files and the scoring are seperated by me. For scoring another script is used by me, which also computes the BERTScore. Therefore, the following is commented out
    """
    if write_rouge:
        scores_dict = scoring.score_outputs(out_path)
        scoring.rouge_log(scores_dict, out_path)
    else:
        print(metrics)
    """


if __name__ == "__main__":
    main()
