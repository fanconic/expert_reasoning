# inference.py
from omegaconf import OmegaConf
import hydra
from src.models.model_module import load_model_and_tokenizer


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    model, tokenizer = load_model_and_tokenizer(cfg)

    SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

    # Prepare an example prompt.
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Calculate pi."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=cfg.model.max_seq_length - cfg.training.max_prompt_length,
    )
    output = (
        model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,
        )[0]
        .outputs[0]
        .text
    )

    print("Inference Output:")
    print(output)


if __name__ == "__main__":
    main()
