# from llm_utils import generate_summarize_prompt_message
import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from llm_utils import (
    get_torch_device,
    list_batches,
    summarize_document,
    clear_accellerator_cache,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

BATCH_SIZE = 2


async def fetch_data(
    url,
):
    async with aiohttp.ClientSession() as session:
        print(url)
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                website_text = soup.get_text()
                # eliminiate multiple empty lines
                website_text = re.sub(r"\n+", "\n", website_text)
                return website_text
            else:
                print(f"Error: {response.status}")
                return None


async def main(args: argparse.Namespace) -> None:

    input_file_path = Path(args.input_json)

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # print(input_file_path)
    # print(output_dir_path)

    torch_device = get_torch_device()

    # print(f"{torch_device=}")

    user_home_dir = Path.home()
    # print(f"{user_home_dir=}")

    llm_model_name = Path(args.llm_model_checkpoint_path)

    # print(f"{llm_model_name=}")

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(str(llm_model_name))
    llm_model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_name),
        device_map=torch_device,
    )
    llm_model.eval()

    site_urls = []
    if not input_file_path.exists() or input_file_path.suffix != ".json":
        raise Exception("json input file does not exist!")

    with open(input_file_path, "r", encoding="utf-8") as fp:
        site_urls = json.load(fp)

    out_file_idx = 0
    for batch in list_batches(site_urls, BATCH_SIZE):
        tasks = [fetch_data(url) for url in batch]
        results = await asyncio.gather(*tasks)

        for result in results:
            site_text_file_name = output_dir_path / Path(f"text_{out_file_idx:04d}.txt")
            summary_file_name = output_dir_path / Path(
                f"text_{out_file_idx:04d}.txt.summary"
            )
            if result:
                with open(site_text_file_name, "w", encoding="utf-8") as fo:
                    fo.write(result)

                _summary, _ = summarize_document(
                    doc_text=result,
                    llm_model=llm_model,
                    tokenizer=tokenizer,
                )
                with open(summary_file_name, "w", encoding="utf-8") as fo:
                    fo.write(_summary)

                clear_accellerator_cache(llm_model.device)
            out_file_idx += 1
        # print(f"{len(results)=}")

    # for site_url in site_urls:
    #     site_text = get_website_text(site_url)
    #     print(f"{site_text=}\n\n\n")

    # print(site_urls)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Crawl the websites URLs specified in the input-json file."
    )

    parser.add_argument(
        "--llm-model-checkpoint-path",
        type=str,
        required=True,
        help="Full Path to the model checkpoint",
    )

    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Fulle Path to the json input file which contains a list of URLs to crawl",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Full Path to the output directory",
    )

    # Parse the arguments passed to the script
    args = parser.parse_args()

    asyncio.run(main(args))

    # example:
    # python crawl_sites_for_text.py --input-json ./rag_data/url_list.json --output-dir ./rag_data/crawled_text/  --llm-model-checkpoint-path  ${HOME}/projects/ai_ml_models/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4/
