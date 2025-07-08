import numpy as np
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainerState, TrainerControl
from vllm import SamplingParams
import wandb
from tqdm import tqdm
import glob
import os


class GenerationEvalCallback(TrainerCallback):
    """
    Every `eval_steps`, run generation on the val set and log GRPO rewards to wandb.
    """

    def __init__(
        self,
        val_dataset,
        tokenizer,
        reward_fns,
        sampling_params: SamplingParams,
        output_dir: str,
        batch_size: int = 1,
    ):
        """
        reward_fns: list of (name, fn) pairs. Each fn(completions, answers, prompts) returns a list of floats.
        """
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.sampling_params = sampling_params
        self.batch_size = batch_size
        self.sums = {name: 0.0 for name, _ in self.reward_fns}
        self.sum_sqs = {name: 0.0 for name, _ in self.reward_fns}
        self.output_dir = output_dir
        

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # only on the process that’s logging
        if state.global_step % args.eval_steps != 0 or not state.is_world_process_zero:
            return

        model = kwargs["model"]
        model.eval()
        
        # fetch the newest adapter - has to be saved on the disk somewhere, cuz god knows why it cannot retrieve it from memory?!
        # Either this, or I am literally too dumb to find the weights in working memory; how did I even get into cambridge?!
        
        # 1) SAVE the in-memory adapter for *this* step before evaluating
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_lora(ckpt_dir)

        # 2) LOAD that freshly saved adapter
        lora_req = model.load_lora(ckpt_dir)


        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda examples: examples,)

        # accumulators
        count = 0

        for batch in tqdm(loader):
                     
            # each batch is a dict with “prompt” and “answer”
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
                lora_request = lora_req,
            )
            gens = [out.outputs[0].text for out in outputs]
            completions = [[{"content": g}] for g in gens]

            # accumulate
            for name, fn in self.reward_fns:
                batch_score = float(np.mean(fn(prompts=prompts, completions=completions, answer=answers)))
                self.sums[name]    += batch_score
                self.sum_sqs[name] += batch_score ** 2
            count += 1
            
        # average and log
        # after looping all batches, compute mean & std
        metrics_mean = {
            f"eval/rewards/{name}/mean": self.sums[name] / count
            for name, _ in self.reward_fns
        }
        metrics_std = {
            f"eval/rewards/{name}/std": np.sqrt(
                (self.sum_sqs[name] / count)
                - (self.sums[name] / count) ** 2
            )
            for name, _ in self.reward_fns
        }

        metrics = {**metrics_mean, **metrics_std, "eval/step": state.global_step}
        wandb.log(metrics)

        # reset for next time
        self.sums    = {name: 0.0 for name, _ in self.reward_fns}
        self.sum_sqs = {name: 0.0 for name, _ in self.reward_fns}
        
        model.train()
