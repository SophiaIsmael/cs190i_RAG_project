"""TODO"""

import re
from typing import Any
from urllib.request import urlopen

import requests
import torch
from bs4 import BeautifulSoup

SUMMARY_WORD_LENGTH = 200


def generate_summarize_prompt_message(
    *,
    text: str,
) -> dict[str, str]:
    """TODO"""

    prompt = f""" You are an agent who can summarize long documents. The document is specified between begin tag: <Document> and end tag </Document>.  

    <Document> 
    {text}
    </Document>

    Summarize the document above in maximum of {SUMMARY_WORD_LENGTH} words.
    """

    return {"role": "user", "content": prompt}


def get_torch_device() -> str:
    """TODO"""
    torch_device = "cpu"
    if torch.cuda.is_available():
        print(f"cuda is available, {torch.cuda.device_count()=}")
        torch_device = "cuda:0"
    elif torch.backends.mps.is_available():
        print("MPS is available!")
        torch_device = "mps"
    else:
        print("No accelerator is available.")

    # torch.set_default_device(torch_device)
    print(f"{torch_device=}")

    return torch_device


def clear_accellerator_cache(device: str) -> None:
    device = str(device)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        # print("cuda cache cleared!")
    elif device.startswith("mps"):
        torch.mps.empty_cache()
        # print("MPS cache cleared!")


def list_batches(input_list, batch_size):
    """TODO"""
    for idx in range(0, len(input_list), batch_size):
        yield input_list[idx : idx + batch_size]


def get_website_text(site_url: str) -> str:
    """TODO"""

    print(site_url)

    response = requests.get(site_url)
    # page = urlopen(site_url)
    # html = page.read().decode("utf-8")
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    website_text = soup.get_text()
    # eliminate multiple empty lines
    return re.sub(r"\n+", "\n", website_text)


def pad_or_trim_hidden_state(
    *,
    context_length: int,
    hidden_state: torch.Tensor,
) -> torch.Tensor:
    """TODO"""
    ctx_tokens, _ = hidden_state.size()

    if ctx_tokens < context_length:
        paddings = [
            torch.zeros_like(hidden_state[1])
            for _ in range(context_length - ctx_tokens)
        ]
        paddings = torch.stack(paddings, dim=0)
        hidden_state = torch.cat([hidden_state, paddings], dim=0)

    hidden_state = hidden_state[0:context_length, :]
    return hidden_state


def summarize_document(
    *,
    llm_model: Any,
    tokenizer: Any,
    doc_text: str,
    max_input_context_length: int = 32_768,
    num_return_sequences: int = 1,
) -> tuple[str, torch.Tensor]:
    """TODO"""
    words = re.findall(r"\w+", doc_text)
    # if len(words) <= SUMMARY_WORD_LENGTH:
    #     return doc_text
    # else:
    with torch.no_grad():
        # use the llm_model to summarize the doc_text
        summarization_prompt = generate_summarize_prompt_message(text=doc_text)

        messages = [summarization_prompt]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
            # padding="max_length",
            truncation=True,
            max_length=max_input_context_length,
        ).to(llm_model.device)

        # conduct text completion
        with torch.no_grad():
            generated_output = llm_model.generate(
                **model_inputs,
                max_new_tokens=1000,
                return_dict_in_generate=True,
                output_hidden_states=True,
                num_return_sequences=num_return_sequences,
            )

            last_hidden_state = []
            output_ids = []
            summaries = []
            for idx in range(num_return_sequences):
                last_hidden_state.append(
                    torch.stack(
                        [
                            token_tensor[-1][idx, -1, :]
                            for token_tensor in generated_output.hidden_states
                        ]
                    ).to("cpu")
                )
                output_ids.append(
                    generated_output.sequences[idx][
                        len(model_inputs.input_ids) :
                    ].tolist()
                )

            for idx in range(num_return_sequences):
                # parsing thinking content
                try:
                    # rindex finding 151668 (</think>) if thinking was enabled
                    index = len(output_ids[idx]) - output_ids[idx][::-1].index(151668)
                except ValueError:
                    index = 0

                # we do not need the thinking_content for summarization
                # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip(
                #     "\n"
                # )
                summaries.append(
                    tokenizer.decode(
                        output_ids[idx][index:], skip_special_tokens=True
                    ).strip("\n")
                )
                
            del generated_output.hidden_states
            del generated_output

        del model_inputs
        del output_ids

        clear_accellerator_cache(device=llm_model.device)
        return summaries, last_hidden_state
