from datasets import load_dataset
from openai import OpenAI
import json
import random
from tqdm import tqdm
import os
from pathlib import Path
import argparse
from evalplus.data import get_mbpp_plus

# Prompts for each dataset
PROMPT = {
    "qasper" : "Please read the paper in the attached PDF and write a peer review using the following template:\n\nPaper Summary*\nDescribe what this paper is about. This should help the program and area chairs to understand the topic of the work and highlight any possible misunderstandings. (Maximum length: 20,000 characters)\n\nSummary of Strengths*\nWhat are the major reasons to publish this paper at a selective *ACL venue? These could include novel and useful methodology, insightful empirical results or theoretical analysis, clear organization of related literature, or any other reason why interested readers of *ACL papers may find the paper useful. (Maximum length: 20,000 characters)\n\nSummary of Weaknesses*\nWhat are the concerns that you have about the paper that would cause you to favor prioritizing other high-quality papers that are also under consideration for publication? Where possible, please number your concerns so authors may respond to them individually. (Maximum length: 20,000 characters)\n\nComments, Suggestions, and Typos*\nIf you have any comments to the authors about how they may improve their paper, other than addressing the concerns above, please list them here. (Maximum length: 20,000 characters)\n\nConfidence*\nChoose one:\n5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work.\n4 = Quite sure. I tried to check the important points carefully.\n3 = Pretty sure, but there's a chance I missed something.\n2 = Willing to defend my evaluation, but I may have missed some details.\n1 = Not my area, or paper is very hard to understand. My evaluation is just an educated guess.\n\nSoundness*\nIs the paper sufficiently sound and thorough? Choose one:\n5 = Excellent: This study is one of the most thorough I have seen, given its type.\n4 = Strong: This study provides sufficient support for all of its claims. Some extra experiments could be nice, but not essential.\n3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.\n2 = Poor: Some of the main claims are not sufficiently supported. There are major technical/methodological problems.\n1 = Major Issues: This study is not yet sufficiently thorough to warrant publication or is not relevant to ACL.\n\nExcitement*\nHow exciting or impactful is this paper? Choose one:\n5 = Highly Exciting: I would recommend this paper to others and/or attend its presentation in a conference.\n4 = Exciting: I would mention this paper to others and/or make an effort to attend its presentation in a conference.\n3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.\n2 = Potentially Interesting: This paper does not resonate with me, but it might with others in the *ACL community.\n1 = Not Exciting: This paper does not resonate with me, and I don't think it would with others in the *ACL community.\n\nOverall Assessment*\nShould this paper be accepted? Choose one:\n5 = Consider for Award: I think this paper could be considered for an outstanding paper award at an *ACL conference.\n4 = Conference: I think this paper could be accepted to an *ACL conference.\n3 = Findings: I think this paper could be accepted to the Findings of the ACL.\n2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.\n1 = Do not resubmit: This paper is so flawed that it has to be fully redone, or it is not relevant to the *ACL community.\n\nOnly output the review without anything else.",
    "mbppplus" : "Please read the problem description in the attached PDF and implement a solution in Python. Only output the source code without anything else.",
    "cnn" : "Please read the article in the attached PDF and write a 2-3 sentence highlight that captures its key point(s). Only output the highlight without anything else."
}

# Getting api key from your environment
API_KEY = os.environ.get('OPENAI_API_KEY')

# Prompting openAI api for each task
def api_prompting(model: str, file_path: str, prompt: str) -> str:
    client = OpenAI(api_key=API_KEY)
    file = client.files.create(file=open(file_path, "rb"), purpose="user_data")
    instruction_message = [
        {"role": "user", "content": [
            {"type": "input_file", "file_id": file.id,},
            {"type": "input_text", "text": prompt,},
        ]},
    ]
    try:
        results = client.responses.create(
            model=MODEL,
            input = instruction_message
        )
        return results.output_text
    except Exception as e:
        return str(e)

def main(input_path: Path, output_path: Path, dataset: str, model: str)->None:
    # Setting task ids and perturbation types for each dataset
    if dataset == "qasper":
        paperqa = load_dataset("allenai/qasper", split="test")
        random.seed(42)
        task_ids = random.sample(range(len(paperqa)), k=101)
        task_ids.remove(357) # This paper is not available
        perturbation_types = ["base", "p1", "p2", "p3", "p4"]
    elif dataset == "mbppplus":
        nl2code = get_mbpp_plus()
        task_ids = []
        for idx, code in nl2code.items():
            task_ids.append(code["task_id"])
        perturbation_types = ["base", "p1", "p2", "p3", "p4", "c1", "c2", "c3", "w1", "w2", "w3", "s1", "s2", "s3"]
    elif dataset == "cnn":
        summary = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        random.seed(42)
        task_ids = random.sample(range(len(summary)), k=300)
        perturbation_types = ["base", "p1", "p2", "p3", "p4"]
    
    # Loading prompt for each dataset
    prompt = PROMPT[dataset]
    for perturbation_type in perturbation_types:
        os.makedirs(output_path / dataset / model, exist_ok=True)
        with open(output_path / dataset /model / f"{perturbation_type}.jsonl", "w") as f:
            # Setting the input file path
            if perturbation_type == "base":
                files_path = input_path / dataset / "original"
            else:
                files_path = input_path / dataset / model / perturbation_type
            
            # Prompting the jobs and Saving the respoonses
            for idx in tqdm(range(len(task_ids))):
                file_path = files_path / f"{idx}.pdf"
                result = api_prompting(model, file_path, prompt)
                if dataset == "mbppplus":
                    json.dump({
                        "idx":idx,
                        "task_id":task_ids[idx],
                        "model":model,
                        "dataset":dataset,
                        "perturbation_type":perturbation_type,
                        "solution":result
                    }, f)
                else:
                    json.dump({
                        "idx":idx,
                        "task_id":task_ids[idx],
                        "model":model,
                        "dataset":dataset,
                        "perturbation_type":perturbation_type,
                        "result":result
                    }, f)
                f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="./pdf_data", type=Path)
    parser.add_argument("--output-path", default="./llm_responses", type=Path)
    parser.add_argument("--dataset", default="qasper", choices=["qasper", "mbppplus", "cnn"], type=str)
    parser.add_argument("--model", default="gpt-4.1", type=str)
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.dataset, args.model)