from datasets import load_dataset
import pymupdf
import arxiv
from tqdm import tqdm
import logging
import json
import random
import os
from evalplus.data import get_mbpp_plus
import argparse
from pathlib import Path

def main(data_path: Path) -> None:
    # Creating repositories
    for dir_name in ["qasper", "mbppplus", "cnn"]:
        os.makedirs(data_path / dir_name / "original", exist_ok = True)
    
    # Downloading original paper pdf in Qasper
    paperqa = load_dataset("allenai/qasper", split="test")
    random.seed(42)
    paperqa_idx = random.sample(range(len(paperqa)), k=101)
    paperqa_idx.remove(357) # This paper is not available
    paperqa = paperqa.select(paperqa_idx)
    logging.basicConfig(level=logging.DEBUG)
    for idx, paperid in enumerate(tqdm(paperqa['id'])):
        paper = next(arxiv.Client(page_size = 1, delay_seconds = 10.0, num_retries = 5).results(arxiv.Search(id_list=[f"{paperid}"])))
        paper.download_pdf(dirpath= data_path / "qasper/original", filename=f"{idx}.pdf")

    # Creating pdf for the MBPP+
    nl2code = get_mbpp_plus()
    nl2code_prompt = []
    for idx, code in enumerate(tqdm(nl2code.values())):
        prompt = code["prompt"]
        doc = pymupdf.open()
        rect = pymupdf.Rect(127, 127, 487, 668)
        page = doc.new_page()
        shape = page.new_shape()
        shape.insert_textbox(
            rect, prompt, fontname="tiro", rotate=0
        )
        shape.finish()
        shape.commit()
        doc.save(data_path / f"mbppplus/original/{idx}.pdf")

    # Creating pdf for the CNN/DailyMail
    summary = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    random.seed(42)
    summary_idx = random.sample(range(len(summary)), k=300)
    summary = summary.select(summary_idx)
    for idx, article in enumerate(summary['article']):
        doc = pymupdf.open()
        rect = pymupdf.Rect(127, 127, 487, 668)
        page = doc.new_page()
        shape = page.new_shape()
        shape.insert_textbox(
            rect, article, fontname="tiro", fontsize=4, rotate=0
        )
        shape.finish()
        shape.commit()
        doc.save(data_path / f"cnn/original/{idx}.pdf")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./pdf_data", type=Path)
    args = parser.parse_args()
    main(args.data_path)