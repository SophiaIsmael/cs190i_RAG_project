from __future__ import annotations

import argparse
import re
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

import torch

from llm_utils import summarize_document, pad_or_trim_hidden_state
from llm_utils import (
    clear_accellerator_cache,
    get_torch_device,
)
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class AutoencoderTrainingDataset(Dataset):
    """
    this class contains the dataset used for the Autoencoder

    Args:
        rag_examples_dir_path (Path): path to the RAG examples
        examples_summary_file_pattern (str):
        tokenizer (AutoTokenizer): tockenizer specifified for the model
        llm_model (AutoModelForCausalLM): model specified for RAG
        max_context_length (int): maximum length of the content
        max_num_samples (int): maximum number of samples
        device (str): device for the model
    """

    def __init__(
        self,
        /,
        *,
        rag_examples_dir_path: Path,
        examples_file_pattern: str,
        tokenizer: AutoTokenizer,
        llm_model: AutoModelForCausalLM,
        max_context_length: int = 1000,
        max_num_samples: int = 10_000,
        device: str = "cpu",
        num_return_sequences: int = 4,
    ):

        super().__init__()

        self.rag_examples_dir_path = Path(rag_examples_dir_path)
        self.autoencoder_example_files = [
            example
            for example in self.rag_examples_dir_path.glob(examples_file_pattern)
        ]

        self.device = device
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.max_context_length = max_context_length

        self._summary_last_hidden_state = []
        # for the autoencoder the output is the same as the input

        self.dataset_size = 0

        self.example_file_ids = {}

        for an_example_file in self.autoencoder_example_files:
            if an_example_file.name not in self.example_file_ids:
                self.example_file_ids[an_example_file.name] = len(self.example_file_ids)

            examples_file_id = torch.tensor(self.example_file_ids[an_example_file.name])

            print(f"tokenizing: {an_example_file}")
            with open(an_example_file, "r", encoding="utf-8") as fp:
                doc_text = fp.read()
                _summaries, _last_hidden_state = summarize_document(
                    llm_model=self.llm_model,
                    tokenizer=self.tokenizer,
                    doc_text=doc_text,
                    num_return_sequences=num_return_sequences,
                )
                for seq_idx in range(num_return_sequences):
                    _hidden_state = pad_or_trim_hidden_state(
                        context_length=self.max_context_length,
                        hidden_state=_last_hidden_state[seq_idx],
                    )
                    # print(f"example summary: {_summary}")
                    # print(f"{last_hidden_state.size()=}")

                    _hidden_state = _hidden_state.to("cpu")

                    # adding num_return_sequences versions of similar document
                    # for contrastive training
                    # print(f"{_hidden_state.size()=}")
                    # print(f"{examples_file_id=}")
                    self._summary_last_hidden_state.append(
                        (examples_file_id, _hidden_state)
                    )

                clear_accellerator_cache(self.device)

            self.dataset_size += 1
            if self.dataset_size >= max_num_samples:
                break

        # print(f"{self._summary_last_hidden_state[0][0]=}")
        # print(f"{self._summary_last_hidden_state[0][1].size()=}")
        print(f"{self.dataset_size=}")

    def __len__(self):
        """returns the number of items"""
        return len(self._summary_last_hidden_state)

    def __getitem__(self, idx):

        # return self._data[idx], self._targets[idx]
        item_id, last_hidden_state = self._summary_last_hidden_state[idx]
        ret_tensor = last_hidden_state.to(self.device)
        item_id = item_id.to(self.device)

        # for an autoencoder, both input and target are the same
        return (item_id, ret_tensor), ret_tensor


# jsut for testing
if __name__ == "__main__":
    torch_device = get_torch_device()

    clear_accellerator_cache(torch_device)

    print(f"{torch_device=}")

    parser = argparse.ArgumentParser(description="Summarize text file.")

    parser.add_argument(
        "--llm-model-checkpoint-path",
        type=str,
        required=True,
        help="Full Path to the model checkpoint",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        required=True,
        help="Fulle Path to dir where the rag text examples are",
    )
    parser.add_argument(
        "--examples-file-pattern",
        type=str,
        required=True,
        help="the file name pattern of summary files",
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

    autoencoder_dataset = AutoencoderTrainingDataset(
        rag_examples_dir_path=args.examples_dir,
        examples_file_pattern=args.examples_file_pattern,
        tokenizer=tokenizer,
        llm_model=llm_model,
        max_num_samples=4,
        device=torch_device,
    )

    test_cycles = 10

    # autoencoder_examples_iter = iter(autoencoder_dataset)
    for tr_exmpl in autoencoder_dataset:
        print(tr_exmpl)
        test_cycles -= 1
        if test_cycles <= 0:
            break

    # example:
    # python autoencoder_dataset.py --llm-model-checkpoint-path  ${HOME}/projects/ai_ml_models/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4/   --examples-dir ./rag_data/crawled_text/  --examples-file-pattern "*.txt"
