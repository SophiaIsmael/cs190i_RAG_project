import argparse
import re
from pathlib import Path

import torch

# from llm_utils import generate_prompt_message, get_instruct_message
from llm_utils import (
    clear_accellerator_cache,
    get_torch_device,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


# from llm_utils import generate_summarize_prompt_message


def main() -> None:

    parser = argparse.ArgumentParser(description="Summarize text file.")

    parser.add_argument(
        "--llm-model-checkpoint-path",
        type=str,
        required=True,
        help="Full Path to the model checkpoint",
    )

    # Parse the arguments passed to the script
    args = parser.parse_args()

    torch_device = get_torch_device()

    llm_model_name = Path(args.llm_model_checkpoint_path)

    print(f"{llm_model_name=}")

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(str(llm_model_name))
    llm_model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_name),
        device_map=torch_device,
    )
    llm_model.eval()

    while True:
        clear_accellerator_cache(torch_device)

        user_prompt = input("type Ctrl-c to exit\nUser: ")
        if user_prompt:
            # print(user_prompt)

            messages = [{"role": "user", "content": user_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)

            # conduct text completion
            with torch.no_grad():
                generated_ids = llm_model.generate(**model_inputs, max_new_tokens=32768)
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

                # parsing thinking content
                try:
                    # rindex finding 151668 (</think>)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                thinking_content = tokenizer.decode(
                    output_ids[:index], skip_special_tokens=True
                ).strip("\n")
                content = tokenizer.decode(
                    output_ids[index:], skip_special_tokens=True
                ).strip("\n")

                print("thinking content:", thinking_content)
                print("\nSystem:", content)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as ex:
        pass


# python plain_chatbot.py --llm-model-checkpoint-path  ${HOME}/projects/ai_ml_models/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4/ 
