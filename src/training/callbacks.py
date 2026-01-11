import numpy as np
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainerState, TrainerControl
from vllm import SamplingParams
import wandb
from tqdm import tqdm
import os
import shutil


class GenerationEvalCallback(TrainerCallback):
    """
    Every `eval_steps`, run generation on the val set and log GRPO rewards to wandb.
    Keeps only the best checkpoint (based on correctness) and the most recent checkpoint.
    """

    def __init__(
        self,
        val_dataset,
        tokenizer,
        reward_fns,
        sampling_params: SamplingParams,
        output_dir: str,
        batch_size: int = 1,
        metric_for_best_model: str = "correctness",  # Name of the reward function to track
        greater_is_better: bool = True,
    ):
        """
        reward_fns: list of (name, fn) pairs. Each fn(completions, answers, prompts) returns a list of floats.
        metric_for_best_model: which reward function to use for determining "best" model
        """
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.sampling_params = sampling_params
        self.batch_size = batch_size
        self.sums = {name: 0.0 for name, _ in self.reward_fns}
        self.sum_sqs = {name: 0.0 for name, _ in self.reward_fns}
        self.output_dir = output_dir
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        
        # Track best model
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.best_checkpoint = None
        self.last_checkpoint = None

    def _is_better(self, current_metric):
        """Check if current metric is better than best metric"""
        if self.greater_is_better:
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric

    def _cleanup_checkpoints(self, keep_checkpoints):
        """Remove all checkpoints except those in keep_checkpoints list"""
        if not os.path.exists(self.output_dir):
            return
            
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if item.startswith("checkpoint-") and os.path.isdir(item_path):
                if item_path not in keep_checkpoints:
                    print(f"Removing checkpoint: {item_path}")
                    shutil.rmtree(item_path)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # only on the process that's logging
        if state.global_step % args.eval_steps != 0 or not state.is_world_process_zero:
            return

        model = kwargs["model"]
        model.eval()

        # 1) SAVE the in-memory adapter for *this* step before evaluating
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_lora(ckpt_dir)
        self.last_checkpoint = ckpt_dir

        # 2) LOAD that freshly saved adapter
        lora_req = model.load_lora(ckpt_dir)

        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda examples: examples,
        )

        # accumulators
        count = 0

        for batch in tqdm(loader):

            # each batch is a dict with "prompt" and "answer"
            prompts = [b["prompt"] for b in batch]  # list of lists of messages
            answers = [b["answer"] for b in batch]  # list of strings

            # turn each prompt into a single string ready for generation
            texts = [
                self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]

            # generate
            outputs = model.fast_generate(
                texts,
                sampling_params=self.sampling_params,
                use_tqdm=False,
                lora_request=lora_req,
            )
            gens = [out.outputs[0].text for out in outputs]
            completions = [[{"content": g}] for g in gens]

            # accumulate
            for name, fn in self.reward_fns:
                batch_score = float(
                    np.mean(
                        fn(prompts=prompts, completions=completions, answer=answers)
                    )
                )
                self.sums[name] += batch_score
                self.sum_sqs[name] += batch_score**2
            count += 1

        # average and log
        # after looping all batches, compute mean & std
        metrics_mean = {
            f"eval/rewards/{name}/mean": self.sums[name] / count
            for name, _ in self.reward_fns
        }
        metrics_std = {
            f"eval/rewards/{name}/std": np.sqrt(
                (self.sum_sqs[name] / count) - (self.sums[name] / count) ** 2
            )
            for name, _ in self.reward_fns
        }

        metrics = {**metrics_mean, **metrics_std, "eval/step": state.global_step}
        
        # Get the metric we're tracking for best model
        current_metric = metrics_mean[f"eval/rewards/{self.metric_for_best_model}/mean"]
        
        # Check if this is the best model so far
        if self._is_better(current_metric):
            self.best_metric = current_metric
            self.best_checkpoint = ckpt_dir
            metrics["eval/best_metric"] = self.best_metric
            metrics["eval/best_step"] = state.global_step
            print(f"New best model at step {state.global_step} with {self.metric_for_best_model}={current_metric:.4f}")
        
        wandb.log(metrics)

        # Clean up old checkpoints, keeping only best and last
        checkpoints_to_keep = []
        if self.best_checkpoint:
            checkpoints_to_keep.append(self.best_checkpoint)
        if self.last_checkpoint and self.last_checkpoint != self.best_checkpoint:
            checkpoints_to_keep.append(self.last_checkpoint)
        
        self._cleanup_checkpoints(checkpoints_to_keep)

        # reset for next time
        self.sums = {name: 0.0 for name, _ in self.reward_fns}
        self.sum_sqs = {name: 0.0 for name, _ in self.reward_fns}

        model.train()

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """At the end of training, copy best model to a well-named directory"""
        if not state.is_world_process_zero:
            return
            
        if self.best_checkpoint:
            best_model_dir = os.path.join(self.output_dir, "best_model")
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)
            shutil.copytree(self.best_checkpoint, best_model_dir)
            print(f"Best model (step {self.best_checkpoint.split('-')[-1]}) saved to {best_model_dir}")
            print(f"Best {self.metric_for_best_model}: {self.best_metric:.4f}")
            
        

class SaveBestByMetricCallback(TrainerCallback):
    def __init__(self, metric_key: str, output_dir: str, greater_is_better: bool = True):
        self.metric_key = metric_key
        self.output_dir = output_dir
        self.greater = greater_is_better
        self.best = None
        self.trainer = None  # will be injected after trainer is created

        os.makedirs(self.output_dir, exist_ok=True)
        self.best_dir = os.path.join(self.output_dir, "best_model")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        
        last_logging = state.log_history[-1] if state.log_history else {}

        val = last_logging[self.metric_key]
        is_best = (
            self.best is None
            or (val > self.best if self.greater else val < self.best)
        )
        if is_best:
            self.best = val
            self.trainer.save_model(self.best_dir)
            print(f"Best model saved to {self.best_dir}")
            print(f"Best {self.metric_key}: {self.best:.4f}")
                
        return control