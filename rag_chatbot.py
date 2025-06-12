import argparse
import re
from pathlib import Path
from typing import Any

from llm_utils import get_torch_device
from trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from chroma_db_utils import ChromaDBHelper
import torch


def get_rag_prompt(user_prompt: str, relevant_documents: list[str]) -> str:

    relevant_doc_tags = [
        f"<DOCUMENT>\n{rel_doc}\n</DOCUMENT>\n" for rel_doc in relevant_documents
    ]
    relevant_doc_tags = "\n".join(relevant_doc_tags)

    rag_prompt = f"""
    You are an agent capable of extracting information from relevant documents to answer a user question.  
    The template below contains a user question between the begin tag: <USER_QUESTION> and end tag: </USER_QUESTION>.  You are also given relevant documents, between the begin tag: <RELEVANT_DOCUMENTS> and the end tag: </RELEVANT_DOCUMENTS>.  Each relevant document is enclosed between the begin tag: <DOCUMENT> and the end tag: </DOCUMENT>.  
    Use the relevant documents below to anser the user question:

    <USER_QUESTION>
    {user_prompt}
    </USER_QUESTION>

    <RELEVANT_DOCUMENTS>
    {relevant_doc_tags}
    </RELEVANT_DOCUMENTS>
    """
    return rag_prompt


def main() -> None:

    parser = argparse.ArgumentParser(description="Summarize text file.")

    parser.add_argument(
        "--llm-model-checkpoint-path",
        type=str,
        required=True,
        help="Full Path to the model checkpoint",
    )
    parser.add_argument(
        "--autoencoder-checkpoint-path",
        type=str,
        required=True,
        help="Fulle Path to dir where the autoencoder checkpoint is",
    )

    parser.add_argument(
        "--chroma-db-dir",
        type=str,
        required=True,
        help="the full path to where chroma_db files are",
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
    llm_model = llm_model.to(torch_device)
    llm_model.eval()

    rag_autoencoder_model, _, _ = Trainer.load_from_checkpoint(
        model_module="autoencoder_model",
        model_class_name="RAGAutoencoder",
        checkpoint_file=args.autoencoder_checkpoint_path,
        device=torch_device,
    )
    rag_autoencoder_model = rag_autoencoder_model.to(torch_device)
    rag_autoencoder_model.eval()

    chromadb_helper = ChromaDBHelper(
        chroma_db_dir=args.chroma_db_dir,
        llm_model=llm_model,
        tokenizer=tokenizer,
        autoencoder_model=rag_autoencoder_model,
        max_summary_tokens=1000,
    )

    while True:
        user_prompt = input("type Ctrl-c to exit\nUser: ")
        if user_prompt:
            print(f"{user_prompt=}")

            res = chromadb_helper.get_neighbors(user_prompt, 4)
            print(res["ids"])

            rag_prompt = get_rag_prompt(
                user_prompt=user_prompt, relevant_documents=res["documents"]
            )

            # print(rag_prompt)

            messages = [{"role": "user", "content": rag_prompt}]
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

            # get embeddings for the user prompt

            # find the nearest k neighbors of the user prompt embeddings

            # append the k-nearest documents to the user prompt
            #   and build an augmented prompt

            # query the model with augmented prompt


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as ex:
        pass


# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# python rag_chatbot.py --llm-model-checkpoint-path  ${HOME}/projects/ai_ml_models/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4/ --autoencoder-checkpoint-path $HOME/projects/cs190i_collaboration/checkpoints/latest.RAGAutoencoder.ckpt  --chroma-db-dir ./rag_data/chroma_db/
