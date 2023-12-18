#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import re
import gc
import sys
from dataclasses import dataclass, field
#HA: the Dict import was not part of the original DANCER code. It has been added for the ismpleRouge_objective function used later
from typing import Optional, Dict

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import torch
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
#HA: removing the import of get_last_checkpoint. It would have been used to resume training from checkpoint_dir, which will not be used by ray
from transformers.trainer_utils import is_main_process #, get_last_checkpoint
#HA: removing the imort of EarlyStoppingCallback, instead the training is relying on population based bandits and the time limit to stop trials
#from transformers import EarlyStoppingCallback

#HA: following imports have been added by me
import ray
from ray import tune
from ray.air.config import CheckpointConfig
from ray.tune import CLIReporter
#HA: for population based bandits
import GPy
import sklearn
from ray.tune.examples.pbt_function import pbt_function
from ray.tune.schedulers.pb2 import PB2
from ray.tune.schedulers.pb2_utils import (normalize, optimize_acq, select_length, UCB, standardize, TV_SquaredExp,)
#HA using my own Trainer
from customTrain import HA_Trainer
#HA: for getting best checkpoint
import glob
import warnings

#HA: for reading trainer_state.json when searching for best chkpt
import json

with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    #HA: added to push the best model to huggingface hub, after the training
    ha_push_to_hub: bool = field(
        default=False,
        metadata = {"help": "Whether or not to push best model checkpoint to huggingface hub."},
    )
    ha_out_name: Optional[str] = field(
        default = None,
        metadata={"help": "Name of saved the model"}
    )
    #HA: added to save memory
    ha_use_cache: bool = field(
        default=False,
        metadata = {"help": "Sets use_cache in model.config"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
            "pegasus) or translation (or translation_{xx}_to_{yy})."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge/sacreblue) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge/sacreblue) on "
            "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    source_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    #HA: path to store ray objectives (trials) in
    storage_path: str = field(
        default="./ray_results",
        metadata = {"help": "Folder to store ray objectives (trials) in."},
    )
    #HA: added
    n_agents: int = field(
        default=1,
        metadata = {"help": "Number of agents/trials/objectives"},
    )
    #HA: added
    training_duration: int = field(
        default=1,
        metadata = {"help": "Duration of training in minutes"},
    )
    #HA: added
    lr_upper: float = field(
        default=5e-3,
        metadata = {"help": "Upper bound for learning rate during hyperparameter search"}
    )
    #HA: added
    lr_lower: float = field(
        default=2e-5,
        metadata = {"help": "Lower bound for learning rate during hyperparameter search"}
    )
    #HA: added
    wd_upper: float = field(
        default=0.01,
        metadata = {"help": "Upper bound for weight decay during hyperparameter search"}
    )
    #HA: added
    wd_lower: float = field(
        default=0.0,
        metadata = {"help": "Lower bound for weight decay during hyperparameter search"}
    )

	
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if not self.task.startswith("summarization") and not self.task.startswith("translation"):
            raise ValueError(
                "`task` should be summarization, summarization_{dataset}, translation or translation_{xx}_to_{yy}."
            )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if model_args.ha_push_to_hub and model_args.ha_out_name == None:
        warnings.warn("ha_push_to_hub is set to True, but no ha_out_name is given. Cannot push the model to hf hub without a name for the model")
        return None
    #HA: The save and evaluation strategy should be set to steps. Using epochs should be seriously reconsidered as explained below
    if training_args.save_strategy != "steps":
        warnings.warn("The code has been tested and executed with steps as save_strategy. Using another strategy could lead to issues with ray tune.")
        return None
    if training_args.evaluation_strategy != "steps":
        warnings.warn("The code has been tested and executed with steps as evaluation_strategy. Using another strategy could lead to issues with ray tune. When completing an epoch, ray cannot continue a trial with the next epoch. The trial will not continue to train, only the process will run until the time limit has been reached")
        return None

    #HA: I highly suggest to have save_steps and eval_steps to match due to the reasons explained below
    if training_args.save_steps != training_args.eval_steps:
        warnings.warn("With ray the eval steps and save steps should match. It is possible to use different values for the parameters, but it can lead to unwanted behavior. The behavior is then dependent on if the number of trial matches the number of used gpus. It can result in a model not continuing from a chekcpoint, but training completely anew.")


    #HA: pushing to hub with multiple trials probably will not work. Not tested, since not needed in my case
    if training_args.push_to_hub == True:
        warnings.warn("Manually deactivating push_to_hub training argument")
    training_args.push_to_hub = False

    #HA: The code would have to be rewritten to work with ray. Since resuming from a checkpoint is not a scenario used by me, it is commented out
    # Detecting last checkpoint.
    #last_checkpoint = None
    #if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #        raise ValueError(
    #            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #            "Use --overwrite_output_dir to overcome."
    #        )
    #    elif last_checkpoint is not None:
    #        logger.info(
    #            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    #HA: added to use tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        #HA: added to set num_beams to 1
        num_beams=data_args.num_beams,
        #HA:max_new_tokens and max_length both controll how long the target sequence is. PEGASUS-X has default of 16384, which leads to very slow generation
        max_new_tokens=data_args.max_target_length,
        max_length=data_args.max_target_length,
        #HA: added to save GPU memory
        gradient_checkpointing=True,
        use_cache=model_args.ha_use_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        config=config,
    )
    #HA: ray does not use a model, but the model_init to initialize a model for each trial, so this can be commented out
    #model = AutoModelForSeq2SeqLM.from_pretrained(
    #    model_args.model_name_or_path,
    #    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #    config=config,
    #    cache_dir=model_args.cache_dir,
    #    revision=model_args.model_revision,
    #    use_auth_token=True if model_args.use_auth_token else None,
    #)
    #HA: the model init function ray tune uses
    def model_init(trial):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config, revision=model_args.model_revision)
        #HA: enabling gradient checkpointing
        model.config.gradient_checkpointing=True
        return model

    #HA: MBartTokenizer is not used by me and the model variable is not getting defined as explained above. Therefore, following section can be commented out
    # Set decoder_start_token_id    
    #if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
    #    assert (
    #        data_args.target_lang is not None and data_args.source_lang is not None
    #    ), "mBart requires --target_lang and --source_lang"
    #    if isinstance(tokenizer, MBartTokenizer):
    #        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
    #    else:
    #        model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    #if model.config.decoder_start_token_id is None:
    #    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if data_args.task.startswith("translation") or isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if data_args.source_lang is not None:
            tokenizer.src_lang = data_args.source_lang
        if data_args.target_lang is not None:
            tokenizer.tgt_lang = data_args.target_lang

    # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
    # them all).
    source_lang, target_lang, text_column, summary_column = None, None, None, None

    if data_args.task.startswith("summarization"):
        # Get the column names for input/target.
        dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
        if data_args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = data_args.text_column
            if text_column not in column_names:
                raise ValueError(
                    f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
                )
        if data_args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = data_args.summary_column
            if summary_column not in column_names:
                raise ValueError(
                    f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
                )
    else:
        # Get the language codes for input/target.
        lang_search = re.match("translation_([a-z]+)_to_([a-z]+)", data_args.task)
        if data_args.source_lang is not None:
            source_lang = data_args.source_lang.split("_")[0]
        else:
            assert (
                lang_search is not None
            ), "Provide a source language via --source_lang or rename your task 'translation_xx_to_yy'."
            source_lang = lang_search.groups()[0]

        if data_args.target_lang is not None:
            target_lang = data_args.target_lang.split("_")[0]
        else:
            assert (
                lang_search is not None
            ), "Provide a target language via --target_lang or rename your task 'translation_xx_to_yy'."
            target_lang = lang_search.groups()[1]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        if data_args.task.startswith("translation"):
            inputs = [ex[source_lang] for ex in examples["translation"]]
            targets = [ex[target_lang] for ex in examples["translation"]]
        else:
            inputs = examples[text_column]
            targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        #HA: added this to set global attention on first token for LED as suggested by Beltagy et al.
        #HA: see https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing
        if model_args.model_name_or_path == "allenai/led-base-16384":
            model_inputs["global_attention_mask"] = len(model_inputs["input_ids"]) * [[0 for _ in range(len(model_inputs["input_ids"][0]))]]
            model_inputs["global_attention_mask"][0][0] = 1
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        #HA: original DANCER placed the train_dataset assignment here and the check afterwards. It should match the order as for eval and test set
        # train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        #HA: as explained above, the order of the assignment of train_dataset has been adjusted by me
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric_name = "rouge" if data_args.task.startswith("summarization") else "sacrebleu"
    metric = load_metric(metric_name)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        if metric_name == "rouge":
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        else:  # sacrebleu
            labels = [[label] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if metric_name == "rouge":
            #HA: trained LED model had learned to produce always the same output regardless of the input. To counteract this, the rouge scores are set to zero if the produced output summaries are too similar to another
            output_similarity = metric.compute(predictions=decoded_preds[1:], references=[decoded_preds[0] for i in range(len(decoded_preds)-1)])
            if output_similarity['rouge2'].mid.fmeasure > 0.5:
                result = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
            else:
                result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
                # Extract a few results from ROUGE
                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}

        #HA: Prediction lengths are not of interest, to speed up the evaluation it is commented out
        #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        #result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    #HA: EarlyStoppingCallback not used by me as mentioned earlier. Instead, training is relying on population based bandits to handle bad performing trials
    # es_callback = EarlyStoppingCallback(early_stopping_patience=3)

    #HA: this is the intial set of parameters, during permutation the hyperparam_bounds in the PB2 class wil be used
    #HA: the hp_space in the hyperparameter_search method cannot be empty, so I provide some values here
    tune_config = {
        "weight_decay": tune.choice([0.0, 0.01]),
    }

    #HA: the objective to be improved by population based bandits.
    #HA: the default objective uses the sum of all metrics of the coput_metrics function. I only want the eval_rouge2 metric
    #HA: used the default objective function of ray tune as orientation
    def simpleRouge_objective(metrics: Dict[str, float]) -> float:
        return metrics['eval_rouge2']

    #HA: reports hyperparmameters and metrics
    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
        },
        #HA: eval_rouge2 and objective should be the same. Kind of a sanity check to report both
        metric_columns=["eval_rouge2", "objective", "eval_rouge1", "eval_rougeL", "eval_rougeLsum", "eval_loss", "eval_runtime"],
        max_report_frequency=60,
    )

    #HA: the algorithm used to improve the objective
    scheduler = ray.tune.schedulers.pb2.PB2(
        #HA: this allows perturbation at every eval step
        time_attr="training_iteration",
        perturbation_interval=1,
        #HA: the metric to optimize. eval_rouge2 could have been used here, but to fit the ray tune framework, I set it to objective
        metric="objective",
        #HA: the direction the objective should have; With max: the higher the objective, the better
        mode="max",
        #the lower and upper bound of the hyperparameters
        hyperparam_bounds={
            "learning_rate": [data_args.lr_lower, data_args.lr_upper],
            "weight_decay": [data_args.wd_lower, data_args.wd_upper]
        },
    )

    trainer = HA_Trainer(
        model_init = model_init,
        args = training_args,
        train_dataset = train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )
    #HA: calling my own trainer class instead of the transoformers trainer
    # Initialize our Trainer
    """
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[es_callback]
    )
    """

    #HA to ensure checkpointing
    trainer.use_tune_checkpoints=True

    #HA: If not set, a warning message appears and suggests to set the env variable. Works without setting it too
    os.environ["TOKENIZERS_PARALLELISM"]="1"

    #HA: running the hyperparameter search
    best_trial = trainer.hyperparameter_search(
        direction="max",
        backend="ray",
        n_trials=data_args.n_agents,
        hp_space=lambda _: tune_config,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_score_attr="objective",
        compute_objective=simpleRouge_objective,
        storage_path=data_args.storage_path,
        time_budget_s=60*data_args.training_duration,
        #HA: checkpoint config does not work reliably, but it does not hurt to have it here
    )

    #HA: The training function is replaced with the population based bandits training, so the original DANCER training code is commented out
    # Training
    #if training_args.do_train:
    #    if last_checkpoint is not None:
    #        checkpoint = last_checkpoint
    #    elif os.path.isdir(model_args.model_name_or_path):
    #        checkpoint = model_args.model_name_or_path
    #    else:
    #        checkpoint = None
    #    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #    trainer.save_model()  # Saves the tokenizer too for easy upload

    #    metrics = train_result.metrics
    #    max_train_samples = (
    #        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    #    )
    #    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    #train_metrics = metrics

    #HA: since the trainer does not get a model, but model_init function, evaluation and prediction will not work like this. Model would have to be loaded first.
    #HA: For my case, there is no value in doing evaluation and prediction after the training anyway, so it will be commented out
    
    # Evaluation
    #if training_args.do_eval:
    #    logger.info("*** Evaluate ***")

    #    metrics = trainer.evaluate(
    #        max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
    #    )
    #    max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
    #    metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

    #eval_metrics = metrics
    #print(eval_metrics)


    #if training_args.do_predict:
    #    logger.info("*** Test ***")

    #    test_results = trainer.predict(
    #        test_dataset,
    #        metric_key_prefix="test",
    #        max_length=data_args.val_max_target_length,
    #        num_beams=data_args.num_beams,
    #    )
    #    metrics = test_results.metrics
    #    max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
    #    metrics["test_samples"] = min(max_test_samples, len(test_dataset))

    #    if trainer.is_world_process_zero():
    #        if training_args.predict_with_generate:
    #            test_preds = tokenizer.batch_decode(
    #                test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #            )
    #            test_preds = [pred.strip() for pred in test_preds]
    #            output_test_preds_file = os.path.join(training_args.output_dir, "test_preds_seq2seq.txt")
    #            with open(output_test_preds_file, "w") as writer:
    #                writer.write("\n".join(test_preds))
                    
    #test_metrics = metrics

    #HA: there is no model defined, so the del should be removed
    del trainer #, model
    gc.collect()
    torch.cuda.empty_cache()

    #HA: the metrics have not been filled with values with my code and are not of interest here, so the statement should be commented out
    #return train_metrics #, eval_metrics, test_metrics

    #HA: getting best checkpoint from the trials
    objectives = list(filter(os.path.isdir, glob.glob(data_args.storage_path+"/_objective_*")))
    objectives.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    objective = objectives[0]

    trials = glob.glob(objective+"/_objective_*")

    #HA: get all checkpoint scores
    score_metric = 'eval_rouge2'
    res = {}

    for trial_dir in trials:
        chkpt0_dirs = glob.glob(trial_dir+"/checkpoint_[0-9]*")
        if len(chkpt0_dirs) == 0:
            continue
        chkpt_dirs = []
        for dir in chkpt0_dirs:
            if len(glob.glob(dir+"/checkpoint*")) > 0:
                chkpt_dirs.append(glob.glob(dir+"/checkpoint*")[0])
        for dir in chkpt_dirs:
            if len(glob.glob(dir+"/trainer_state.json")) == 0:
                continue
            with open(os.path.join(dir, 'trainer_state.json'), 'r') as f:
                try:
                    contents = json.loads(f.read())
                    res[dir] = contents['log_history'][-1][score_metric]
                except KeyError:
                    print('checkpoint does not have eval_rouge2 at last log_history entry, at checkpoint:', dir)
    r_max = max(res.values())

    #HA: getting checkpoint with the highest score
    chkpts = []
    for key in res:
        if res[key] == r_max:
            chkpts.append(key)

    chkptForHfHub = ''
    #HA: if there is only 1 checkpoint with the highest score, then use it. If there are multiple checkpoints with the same score, use the checkpoint from the earliest step (across different trials).
    #HA: if there are multiple trials with the same rouge score (the best one) at the same step in training, it uses the checkpoint that was first read (probably the one with a lower trial number).
    #HA: it is an unlikely scenario, unless the best checkpoint has been copied to other trials. I would suggest checking the files manually then to ensure that everything worked correctly
    if len(chkpts) < 1:
        print("There has been an issue with getting the best checkpoint. Please check the files manually")
    elif len(chkpts) == 1:
        chkptForHub = chkpts[0]
    else:
        chkptsSplit = []
        for chkpt in chkpts:
            chkptsSplit.append(os.path.split(chkpt))
        chkptsSplit = dict(chkptsSplit)
        firstBestChkpt = min(chkptsSplit.values())
        firstBestChkptDir = []

        for key in chkptsSplit:
            if chkptsSplit[key] == firstBestChkpt:
                firstBestChkptDir.append(key)
        if len(firstBestChkptDir) < 1:
            print("There has been an issue with getting the best checkpoint. Please check the files manually")
        elif len(firstBestChkpt) == 1:
            print("There have been multiple checkpoints with the same rouge2 score. Choosing first checkpoint:", firstBestChkpt[0])
            print("See other checkpoints with the same score:", chkpts)
            chkptForHfHub = firstBestChkpt[0]
        else:
            print("Multiple trials have the same max rouge score at the same step. Choosing checkpoint:", firstBestChkpt[0])
            print("see the other checkpoints with the same score:", chkpts)
            chkptForHfHub = firstBestChkpt[0]

    if chkptForHub and model_args.ha_push_to_hub:
        model = AutoModelForSeq2SeqLM.from_pretrained(chkptForHub)
        model.push_to_hub(model_args.ha_out_name)
            
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
