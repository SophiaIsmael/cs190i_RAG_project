import argparse
import time
from pathlib import Path
from typing import Any

import chromadb
import torch
from autoencoder_model import RAGAutoencoder
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from llm_utils import (
    clear_accellerator_cache,
    get_torch_device,
    summarize_document,
    pad_or_trim_hidden_state,
)
from trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer

RAG_COLLECTION_NAME = "rag_collection"


class ChromaDBHelper:
    def __init__(
        self,
        chroma_db_dir: str,
        llm_model: Any,
        tokenizer: Any,
        autoencoder_model: RAGAutoencoder,
        max_summary_tokens: int = 1000,
    ):
        self.chroma_db_dir = chroma_db_dir
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.autoencoder_model = autoencoder_model
        self.max_summary_tokens = max_summary_tokens

        # ensure chroma db dir exists
        Path(self.chroma_db_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.rag_embedding_fn = RAGEmbeddingFunction(
            llm_model=llm_model,
            tokenizer=tokenizer,
            autoencoder_model=self.autoencoder_model,
            max_summary_tokens=1000,
        )

    def add_rag_documents(
        self,
        document_text: str,
        document_id: str,
    ):
        client = chromadb.PersistentClient(path=self.chroma_db_dir)
        collection = client.get_or_create_collection(
            name=RAG_COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    # "embedding_function": None,
                    # "ef_search": 100,
                    # "ef_construction": 100,
                    # "max_neighbors": 16,
                    # "num_threads": 4,
                },
                "embedding_function": self.rag_embedding_fn,
            },
            embedding_function=self.rag_embedding_fn,
        )  # Get a collection object from an existing collection, by name. If it doesn't exist, create it.

        collection.add(
            documents=[document_text],
            # metadatas={},
            ids=[document_id],
        )

    def get_neighbors(
        self,
        query_document_text: str,
        num_neighbors: int,
    ) -> list[str]:
        """TODO"""

        client = chromadb.PersistentClient(path=self.chroma_db_dir)
        collection = client.get_or_create_collection(
            name=RAG_COLLECTION_NAME,
            embedding_function=self.rag_embedding_fn,
        )  # Get a collection object from an existing collection, by name. If it doesn't exist, create it.

        query_res = collection.query(
            query_texts=[query_document_text],
            n_results=num_neighbors,
            include=["documents"],
            # where={"metadata_field": "is_equal_to_this"},
            # where_document={"$contains":"search_string"},
            # ids=["id1", "id2", ...]
        )

        return query_res


class RAGEmbeddingFunction(EmbeddingFunction[Documents]):
    """TODO"""

    def __init__(
        self,
        /,
        *,
        llm_model: Any,
        tokenizer: Any,
        autoencoder_model: RAGAutoencoder,
        max_summary_tokens: int,
    ):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.autoencoder_model = autoencoder_model
        self.max_summary_tokens = max_summary_tokens

    def __call__(self, input: Documents) -> Embeddings:
        """TODO"""
        ret_embeddings = []

        with torch.no_grad():
            for document_text in input:
                _summary, last_hidden_state = summarize_document(
                    llm_model=self.llm_model,
                    tokenizer=self.tokenizer,
                    doc_text=document_text,
                )

                last_hidden_state = pad_or_trim_hidden_state(
                    context_length=self.max_summary_tokens,
                    hidden_state=last_hidden_state[0],
                )
                last_hidden_state = torch.unsqueeze(last_hidden_state, dim=0).to(
                    self.llm_model.device
                )

                _input_id = torch.tensor(0, device=self.llm_model.device)
                embeddings, output = self.autoencoder_model((_input_id, last_hidden_state))
                embeddings = torch.squeeze(embeddings, 0)
                embeddings = embeddings.to("cpu").detach()

                ret_embeddings.append(embeddings.tolist())
                del embeddings
                del output
                clear_accellerator_cache(self.llm_model.device)

        return ret_embeddings


if __name__ == "__main__":

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
        "--examples-dir",
        type=str,
        required=True,
        help="the full path to where rag docs are",
    )

    parser.add_argument(
        "--chroma-db-dir",
        type=str,
        required=True,
        help="the full path to where chroma_db files are",
    )
    parser.add_argument(
        "--examples-file-pattern",
        type=str,
        required=False,
        default="*.txt",
        help="the file name pattern of document files",
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

    docs_list = list(Path(args.examples_dir).glob(args.examples_file_pattern))

    for idx, doc_name in enumerate(docs_list):
        with open(doc_name, "r", encoding="utf-8") as fp:
            document_text = fp.read()
            chromadb_helper.add_rag_documents(
                document_text=document_text,
                document_id=doc_name.name,
            )
            print(f"{idx=}")
        # time.sleep(5)

# python chroma_db_utils.py --llm-model-checkpoint-path  ${HOME}/projects/ai_ml_models/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4/ --autoencoder-checkpoint-path $HOME/projects/cs190i_collaboration/checkpoints/latest.RAGAutoencoder.ckpt  --examples-dir ./rag_data/crawled_text/  --chroma-db-dir ./rag_data/chroma_db/
