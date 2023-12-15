#HA: these imports are for overwriting the transformers functions
from transformers.trainer_utils import (
    HPSearchBackend,
    BestRun,
    EvaluationStrategy,
    PREFIX_CHECKPOINT_DIR,
    IntervalStrategy
)
from transformers.integrations import (TensorBoardCallback)
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_sigopt,
    run_hp_search_ray,
    run_hp_search_wandb,
    is_ray_tune_available,
)
from transformers.trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from transformers.training_args import ParallelMode  # noqa: E402
from transformers.utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available, is_datasets_available  # noqa: E402
import importlib.util

#HA: used to write own Trainer
from transformers import Seq2SeqTrainer

#HA: for convert_state_flType function
import numpy

import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from transformers.trainer_utils import (ShardedDDPOption)

from transformers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    get_full_repo_name,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

#HA: for save_to_json function overwrite, these imports are needed
import dataclasses
import json

#HA: copied hp_search_ray function from transformers trainer https://github.com/huggingface/transformers/blob/v4.29-release/src/transformers/trainer.py
#HA: changes are marked by comments
def my_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import ray

    def _objective(trial, local_trainer, checkpoint_dir=None):
        try:
            from transformers.utils.notebook import NotebookProgressCallback

            if local_trainer.pop_callback(NotebookProgressCallback):
                local_trainer.add_callback(ProgressCallback)
        except ModuleNotFoundError:
            pass

        checkpoint = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint = os.path.join(checkpoint_dir, subdir)
        local_trainer.objective = None

        local_trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        #HA: to ensure checkpointing at end, remove the if condition
        #if getattr(local_trainer, "objective", None) is None:
        metrics = local_trainer.evaluate()
        local_trainer.objective = local_trainer.compute_objective(metrics)
        #HA: calling the overwrite of _tune_save_checkpoint of my trainer class
        local_trainer._tune_save_checkpoint()
        # HA: objective is no argument of ray.tune.report, it is probably outdated. 
        #ray.tune.report(objective=local_trainer.objective, **metrics, done=True)
        ray.tune.report(**metrics, done=True)

    if not trainer._memory_tracker.skip_memory_metrics:
        from .trainer_utils import TrainerMemoryTracker

        logger.warning(
            "Memory tracking for your Trainer is currently "
            "enabled. Automatically disabling the memory tracker "
            "since the memory tracker is not serializable."
        )
        trainer._memory_tracker = TrainerMemoryTracker(skip_memory_metrics=True)

    # The model and TensorBoard writer do not pickle so we have to remove them (if they exists)
    # while doing the ray hp search.
    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None

    # Setup default `resources_per_trial`.
    if "resources_per_trial" not in kwargs:
        # Default to 1 CPU and 1 GPU (if applicable) per trial.
        kwargs["resources_per_trial"] = {"cpu": 1}
        if trainer.args.n_gpu > 0:
            kwargs["resources_per_trial"]["gpu"] = 1
        resource_msg = "1 CPU" + (" and 1 GPU" if trainer.args.n_gpu > 0 else "")
        logger.info(
            "No `resources_per_trial` arg was passed into "
            "`hyperparameter_search`. Setting it to a default value "
            f"of {resource_msg} for each trial."
        )
    # Make sure each trainer only uses GPUs that were allocated per trial.
    gpus_per_trial = kwargs["resources_per_trial"].get("gpu", 0)
    trainer.args._n_gpu = gpus_per_trial

    # Setup default `progress_reporter`.
    if "progress_reporter" not in kwargs:
        from ray.tune import CLIReporter

        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])
    if "keep_checkpoints_num" in kwargs and kwargs["keep_checkpoints_num"] > 0:
        # `keep_checkpoints_num=0` would disabled checkpointing
        trainer.use_tune_checkpoints = True
        if kwargs["keep_checkpoints_num"] > 1:
            logger.warning(
                f"Currently keeping {kwargs['keep_checkpoints_num']} checkpoints for each trial. "
                "Checkpoints are usually huge, "
                "consider setting `keep_checkpoints_num=1`."
            )
    if "scheduler" in kwargs:
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining

        # Check if checkpointing is enabled for PopulationBasedTraining
        if isinstance(kwargs["scheduler"], PopulationBasedTraining):
            if not trainer.use_tune_checkpoints:
                logger.warning(
                    "You are using PopulationBasedTraining but you haven't enabled checkpointing. "
                    "This means your trials will train from scratch everytime they are exploiting "
                    "new configurations. Consider enabling checkpointing by passing "
                    "`keep_checkpoints_num=1` as an additional argument to `Trainer.hyperparameter_search`."
                )

        # Check for `do_eval` and `eval_during_training` for schedulers that require intermediate reporting.
        if isinstance(
            kwargs["scheduler"], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
        ) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == IntervalStrategy.NO):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    trainable = ray.tune.with_parameters(_objective, local_trainer=trainer)

    @functools.wraps(trainable)
    def dynamic_modules_import_trainable(*args, **kwargs):
        """
        Wrapper around `tune.with_parameters` to ensure datasets_modules are loaded on each Actor.

        Without this, an ImportError will be thrown. See https://github.com/huggingface/transformers/issues/11565.

        Assumes that `_objective`, defined above, is a function.
        """
        if is_datasets_available():
            import datasets.load

            dynamic_modules_path = os.path.join(datasets.load.init_dynamic_modules(), "__init__.py")
            # load dynamic_modules from path
            spec = importlib.util.spec_from_file_location("datasets_modules", dynamic_modules_path)
            datasets_modules = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = datasets_modules
            spec.loader.exec_module(datasets_modules)
        return trainable(*args, **kwargs)

    # special attr set by tune.with_parameters
    if hasattr(trainable, "__mixins__"):
        dynamic_modules_import_trainable.__mixins__ = trainable.__mixins__

    analysis = ray.tune.run(
        dynamic_modules_import_trainable,
        config=trainer.hp_space(None),
        num_samples=n_trials,
        **kwargs,
    )
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3], scope=trainer.args.ray_scope)
    #HA: getting bestrun does not work reliably, not necessary due to my custom logging anyway
    #best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config, analysis)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    #HA: originally returned best_run, but as stated above the trial data is returned instead
    return best_trial

#HA: when trying to save the trainer state, some of the trial hyperparameters are of type numpy.float32, which is not JSON serializable
#HA: therefore, the types must be converted
def convert_state_flType(state):
    for key in state['trial_params'].keys():
        if isinstance(state['trial_params'][key], numpy.float32):
            state['trial_params'][key] = float(state['trial_params'][key])
    return state

#HA: keeping a certain number of checkpoints with ray tune and transformers trainer does not work
#HA: the function will delete every checkpoint except the best. If there are two checkpoints with the same rouge score, then both will be kept
def ha_clean_chkpts(chkpt_dir):
  trial_dir = os.path.split(os.path.split(chkpt_dir)[0])[0]
  # use [0-9] to exclude temporary checkpoints which are named something like checkpoint_tmp[a-z0-9]*
  chkpt0_dirs = glob.glob(trial_dir+"/checkpoint_[0-9]*")
  if len(chkpt0_dirs) < 2:
    return None
  else:
    chkpt_dirs = []
    for dir in chkpt0_dirs:
      if len(glob.glob(dir+"/checkpoint*")) > 0:
        chkpt_dirs.append(glob.glob(dir+"/checkpoint*")[0])

  r_res = {}
  for dir in chkpt_dirs:
    if len(glob.glob(os.path.join(dir, "trainer_state.json"))) == 0:
        continue
    with open(os.path.join(dir, 'trainer_state.json'), 'r') as f:
      try:
        contents = json.loads(f.read())
        r_res[dir] = contents['log_history'][-1]['eval_rouge2']
      except:
        print('latest entry in trainer_state.json log does not have eval_rouge2. See checkpoint: ', dir)
    max_r = max(r_res.values())
    max_chkpts = [dir for dir in r_res if r_res[dir] == max_r]

    #HA: chkpt_dir may include directory which does not have a trainer_state.json yet. Then we would not want to delete the folder. If the directory is in r_res it does have a trainer_state.json. Thats why r_res.keys() is used instead of chkpt_dirs.keys()
    for dir in r_res.keys():
      if dir not in max_chkpts:
        print("deleting: ", dir)
        print("which had score of: ", r_res[dir])
        os.system('rm -r '+ os.path.split(dir)[0])

#HA: creating a new trainer class to controll checkpointing and call  my_hp_search_ray function
class HA_Trainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)

    def _tune_save_checkpoint(self):
        from ray import tune

        #HA: Enforcing checkpointing
        #if not self.use_tune_checkpoints:
        #    return
        with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
            output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            #HA: to only keep the checkpoints with the best eval_rouge2 score, the following funtion will delete worse checkpoints
            #HA: the function is called before saving the new checkpoint. After saving it, there will at most be checkpoints with two different eval_rouge2 scores per trial
            ha_clean_chkpts(output_dir)
            self.save_model(output_dir, _internal_call=True)
            if self.args.should_save:

                #self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                #HA: the original code used the line above, but the trial_params of the trainer's state are of type numpy.float32, which is not json serializable. Therefore the types must be converted with the function convert_state_flType function before saving it to json
                #HA: it follows the implementation of the save_to_json function. The only addition is the the conversion function
                json_string = json.dumps(convert_state_flType(dataclasses.asdict(self.state)), indent=2, sort_keys=True) + "\n"
                with open(os.path.join(output_dir, TRAINER_STATE_NAME), "w", encoding="utf-8") as f:
                    f.write(json_string)

                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def _save_checkpoint(self, model, trial, metrics=None):
        #HA: the _save_checkpoint function is called, when the number of trials and the number of GPUs matches, meaning that there won't be any pending trials
        #HA: if they don't match, the function won't be called anyway
        #HA: but if it's called it generates checkpoints, while keeping only 2 checkpoints, which are not necessarily the best 2. Instead, checkpointing is done with the _tune_save_checkpoint function above, so the function here is emptied out
        pass

    def hyperparameter_search(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ) -> BestRun:
        if backend is None:
            backend = default_hp_search_backend()
            if backend is None:
                raise RuntimeError(
                    "At least one of optuna or ray should be installed. "
                    "To install optuna run `pip install optuna`. "
                    "To install ray run `pip install ray[tune]`. "
                    "To install sigopt run `pip install sigopt`."
                )
        backend = HPSearchBackend(backend)
        if backend == HPSearchBackend.OPTUNA and not is_optuna_available():
            raise RuntimeError("You picked the optuna backend, but it is not installed. Use `pip install optuna`.")
        if backend == HPSearchBackend.RAY and not is_ray_tune_available():
            raise RuntimeError(
                "You picked the Ray Tune backend, but it is not installed. Use `pip install 'ray[tune]'`."
            )
        if backend == HPSearchBackend.SIGOPT and not is_sigopt_available():
            raise RuntimeError("You picked the sigopt backend, but it is not installed. Use `pip install sigopt`.")
        if backend == HPSearchBackend.WANDB and not is_wandb_available():
            raise RuntimeError("You picked the wandb backend, but it is not installed. Use `pip install wandb`.")
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = default_hp_space[backend] if hp_space is None else hp_space
        self.hp_name = hp_name
        self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

        backend_dict = {
            HPSearchBackend.OPTUNA: run_hp_search_optuna,
            #HA: replaced the run_hp_search_ray with my own function
            HPSearchBackend.RAY: my_hp_search_ray,
            HPSearchBackend.SIGOPT: run_hp_search_sigopt,
            HPSearchBackend.WANDB: run_hp_search_wandb,
        }
        best_run = backend_dict[backend](self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run
