"""TODO"""

import argparse
from pathlib import Path

import torch
from autoencoder_dataset import AutoencoderTrainingDataset
from autoencoder_model import RAGAutoencoder
from llm_utils import get_torch_device
from torch.utils.data import DataLoader
from trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train autoencoder model.")

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
        help="the file name pattern of example files",
    )

    parser.add_argument(
        "--num-train-epochs",
        type=int,
        required=False,
        default=1000,
        help="num_train_epochs",
    )

    parser.add_argument(
        "--num-train-batches",
        type=int,
        required=False,
        default=200,
        help="num_train_batches",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=1,
        help="batch_size",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="checkpoint_dir",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="log_dir",
    )

    parser.add_argument(
        "--keep-only-last-checkpoints",
        type=int,
        required=False,
        default=3,
        help="keep_only_last_checkpoints",
    )
    parser.add_argument(
        "--save-checkpoint-every-epochs",
        type=int,
        required=False,
        default=10,
        help="save_checkpoint_every_epochs",
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
    llm_model.to(torch_device)
    llm_model.eval()

    autoencoder_dataset = AutoencoderTrainingDataset(
        rag_examples_dir_path=args.examples_dir,
        examples_file_pattern=args.examples_file_pattern,
        tokenizer=tokenizer,
        llm_model=llm_model,
        max_num_samples=args.num_train_batches,
        device=torch_device,
        num_return_sequences=4,
    )

    # print(f"{len(train_dataset)=}")

    # img_tensor, annotations = train_dataset[0]
    # print(f"{img_tensor.size()=}, {annotations=}")

    train_dataloader = DataLoader(
        autoencoder_dataset,
        batch_size=args.batch_size,
        shuffle=True, # reshffle to utilize dataset randomly
        pin_memory=False,
        generator=torch.Generator(),
        num_workers=0,
        # collate_fn=COCODataset.collate,???
    )

    model = RAGAutoencoder(
        token_embedding_dim=llm_model.config.hidden_size,
        max_context_length=1000,
        device=torch_device,
    )
    model.to(torch_device)

    #  create a trainer:
    trainer = Trainer(
        model=model,
        num_training_batches=args.num_train_batches,
        train_dataloader=train_dataloader,
        checkpoint_dir=args.checkpoint_dir,
        keep_only_last_checkpoints=args.keep_only_last_checkpoints,
        save_checkpoint_every_epochs=args.save_checkpoint_every_epochs,
        log_dir=args.log_dir,
    )

    trainer.perform_train_cycle(
        epochs=args.num_train_epochs,
    )

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# python start_training.py  --llm-model-checkpoint-path  ${HOME}/projects/ai_ml_models/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4/   --examples-dir ./rag_data/crawled_text/  --examples-file-pattern "*.txt" --checkpoint-dir $HOME/projects/cs190i_collaboration/checkpoints --log-dir $HOME/projects/cs190i_collaboration/logs
