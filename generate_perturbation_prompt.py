from datasets import load_dataset
from evalplus.data import get_mbpp_plus
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import random
import os

# Prompt templates for each perturbation types
PROMPT_TEMPLATES = {
    "p2": "your job is modifying each sentence of the following text by introducing hallucinations such that the overall length and structure appear similar to the original, but the actual content and numerical details differ noticeably upon close reading.\n",
    "c1":"Choose at most two words in the sentence, and change them so that they have typos.",
    "c2":"Change at most two letters in the sentence.",
    "c3":"Add at most two extraneous characters to the end of the sentence.",
    "w1":"Replace at most two words in the sentence with synonyms.",
    "w2":"Choose at most two words in the sentence that do not contribute to the meaning of the sentence and delete them.",
    "w3":"Add at most two semantically neutral words to the sentence.",
    "s1":"Add a randomly generated short meaningless handle after the sentence, such as @fasuv3.",
    "s2":"Paraphrase the sentence.",
    "s3":"Change the syntactic structure of the sentence.",
}

def make_prompt(dataset: str, input_type: str, target: str):
    prompts = {}
    
    # MBPP+ requires prompts for the PromptAttack baseline
    if dataset == "mbppplus":
        for key, template in PROMPT_TEMPLATES.items():
            original_input = f"Given a {input_type}, "
            attack_objective = f"your task is to generate a new {input_type} which must satisfy the following conditions: \n"
            attack_objective += f"Keeping the semantic meaning of the new {input_type} unchanged; \n"
            attack_guidance = f"You can finish the task by modifying {input_type} using the following guidance: \n"
            attack_guidance += f"{template} \n"
            attack_guidance += f"Only output the new {input_type} without anything else."
            prompt = original_input + attack_objective + attack_guidance + "\n" + f"{target} ->"
            prompts[key] = prompt

    # Prompt for making hallucination
    prompts["p2"] = f"Given a {input_type}, "+ PROMPT_TEMPLATES["p2"] + f"Only output the new {input_type} without anything else." + "\n" + f"{target} ->"
    return prompts

def load_data(name: str)->list[str]:
    # Loading Qasper dataset
    if name == "qasper":
        paperqa = load_dataset("allenai/qasper", split="test")
        random.seed(42)
        paperqa_idx = random.sample(range(len(paperqa)), k=101)
        paperqa_idx.remove(357)
        paperqa = paperqa.select(paperqa_idx)
        target_texts = []
        for paper in paperqa:
            sections = ["\n".join(section) for section in paper["full_text"]["paragraphs"]]
            if paper["full_text"]["section_name"][0] is None:
                paper["full_text"]["section_name"].pop(0)
                paper["full_text"]["paragraphs"].pop(0)
            full_paper = paper["title"]+"\n"+"Abstract"+"\n"+paper["abstract"]+"\n".join([paper["full_text"]["section_name"][i].replace("::: ","")+"\n"+sections[i] for i in range(len(paper["full_text"]["paragraphs"]))])
            target_texts.append(full_paper)
            
    # Loading MBPP+ dataset
    elif name == "mbppplus":
        nl2code = get_mbpp_plus()
        target_texts = [i["prompt"] for i in nl2code.values()]
        
    # Loading CNN/DailyMail dataset
    elif name == "cnn":
        summary  = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        random.seed(42)
        summary_idx = random.sample(range(len(summary)), k=300)
        summary = summary.select(summary_idx)
        target_texts = list(summary['article'])
        
    # Other datasets are not implemented yet.
    else:
        raise ValueError(f"Invalid dataset name: {name}.")
    return target_texts

def main(prompt_path: Path, dataset: str) -> None:
    # Setting input types for the dataset
    if dataset == "qasper":
        input_type = "paper"
    elif dataset == "mbppplus":
        input_type = "natural language description"
    elif dataset == "cnn":
        input_type = "news article"
    else:
        raise ValueError(f"Invalid dataset name: {dataset}.")
    
    # Loading the data to perturb
    target_texts = load_data(dataset)
    
    # Creating prompts for perturbation
    prompts = [make_prompt(dataset, input_type, text) for text in target_texts]
    
    # Saving prompts for perturbation
    os.makedirs(prompt_path / dataset, exist_ok=True)
    with open(prompt_path / f"{dataset}/prompt.jsonl", 'w') as f:
        for prompt in prompts:
            json.dump(prompt, f)
            f.write("\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-path", default="./perturbation_prompt", type=Path)
    parser.add_argument("--dataset", default="qasper", choices=["qasper", "mbppplus", "cnn"], type=str)
    args = parser.parse_args()
    main(args.prompt_path, args.dataset)