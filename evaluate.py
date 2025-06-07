from datasets import load_dataset
import json
import os
from pathlib import Path
import argparse
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score
from tqdm import tqdm
from transformers import logging
import random
logging.set_verbosity_error()

def main(input_path: Path, sanitized_code_path: Path, dataset: str, model: str) -> None:
    if dataset == "mbppplus":
        perturbation_types = ["c1", "c2", "c3", "w1", "w2", "w3", "s1", "s2", "s3", "p1", "p2", "p3", "p4", "base"]
        for perturbation_type in perturbation_types:
            with open(input_path / dataset / model / f"{perturbation_type}.jsonl") as f:
                responses=[json.loads(i) for i in f.readlines()]
            pattern = re.compile(
                r'''```(?:python[ \t]*\n|\n)(.*?)```''',
                re.IGNORECASE | re.DOTALL | re.VERBOSE
            )
            sanitized_codes = []
            for response in responses:
                if "```" in response["solution"]:
                    sanitized_code = "\n".join([code_block.strip() for code_block in pattern.findall(response["solution"])])
                else:
                    sanitized_code = response["solution"]
                response["solution"] = sanitized_code
                sanitized_codes.append(response)
                
            os.makedirs(sanitized_code_path / dataset / model / "json", exist_ok=True)
            json_file_path = sanitized_code_path / dataset / model / "json" / f"{perturbation_type}.jsonl"
            with open(json_file_path, 'w') as f:
                for sanitized_code in sanitized_codes:
                    json.dump(sanitized_code, f)
                    f.write("\n")
            print(f"dataset: {dataset}, model: {model}, perturbation_type: {perturbation_type}")
            os.system(f"evalplus.evaluate --dataset mbpp --samples {json_file_path}")

            os.makedirs(sanitized_code_path / dataset / model / "code" / perturbation_type, exist_ok=True)
            for sanitized_code in sanitized_codes:
                with open(sanitized_code_path / dataset / model / "code" / perturbation_type / f"{sanitized_code['idx']}.py", "w") as f:
                    f.write(sanitized_code["solution"])
    else:
        if dataset == "cnn":
            summary  = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
            random.seed(42)
            summary_idx = random.sample(range(len(summary)), k=300)
            summary = summary.select(summary_idx)
            ground_truth = list(summary['highlights'])
            perturbation_types = ["p1", "p2", "p3", "p4", "base"]
        else:
            with open(input_path / dataset / model / f"base.jsonl") as f:
                ground_truth = [json.loads(i)["result"] for i in f.readlines()]
            perturbation_types = ["p1", "p2", "p3", "p4"]

        rouge = Rouge()
        for perturbation_type in perturbation_types:
            with open(input_path / dataset / model / f"{perturbation_type}.jsonl") as f:
                responses = [json.loads(i)["result"] for i in f.readlines()]
            assert len(responses) == len(ground_truth)
            bleu_scores = []
            rouge_scores = []
            bert_scores = []
            for i in tqdm(range(len(responses))):
                reference = ground_truth[i]
                model_out = responses[i]
                weights = [
                    (1.,),
                    (1./2., 1./2.),
                    (1./3., 1./3., 1./3.),
                    (1./4., 1./4., 1./4., 1./4.)
                ]
                bleu_score = sentence_bleu([reference.split()], model_out.split(), weights=weights)
                rouge_score = rouge.get_scores(model_out, reference, avg=True)
                bert_score = score([model_out], [reference], lang="en")
                bleu_scores.append(bleu_score)
                rouge_scores.append(rouge_score)
                bert_scores.append(bert_score[2])
            print(f"dataset: {dataset}, model: {model}, perturbation_type: {perturbation_type}")
            for ngram in range(4):
                print(f"bleu_score_{ngram}: {sum([bleu_score[ngram] for bleu_score in bleu_scores])/len(bleu_scores)}")
            for idx in ["rouge-1", "rouge-2", "rouge-l"]:
                print(f"rouge_score_{idx[-1]}: {sum([rouge_score[idx]['f'] for rouge_score in rouge_scores])/len(rouge_scores)}")
            print("bert_scores", sum(bert_scores)/len(bert_scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="./llm_responses", type=Path)
    parser.add_argument("--sanitized-code-path", default="./sanitized_codes", type=Path) # only for mbppplus
    parser.add_argument("--dataset", default="qasper", choices=["qasper", "mbppplus", "cnn"], type=str)
    parser.add_argument("--model", default="gpt-4.1", type=str)
    args = parser.parse_args()
    main(args.input_path, args.sanitized_code_path, args.dataset, args.model)