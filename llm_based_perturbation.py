from openai import OpenAI
import json
import random
from tqdm import tqdm
import os
from pathlib import Path
import argparse

# Getting api key from your environment
API_KEY = os.environ.get('OPENAI_API_KEY')

# Prompting openAI api to perturb the original text
def api_prompting(model: str, prompt: str) -> str:
    instruction_message=[
        {"role": "user", "content": prompt},
    ]
    client = OpenAI(api_key=API_KEY)
    try:
        results = client.chat.completions.create(
            model = model,
            messages = instruction_message
        )
        return results.choices[0].message.content
    except Exception as e:
        return str(e)

def main(prompt_path: Path, output_path: Path, dataset: str, model: str):
    # Getting prompts
    with open(prompt_path / dataset / "prompt.jsonl") as f:
        prompts = [json.loads(j) for j in f.readlines()]

    prompt_types = list(prompts[0].keys())
    os.makedirs(output_path / dataset / model, exist_ok=True)
    for prompt_type in prompt_types:
        # Saving the respoonses from api
        with open(output_path / dataset / model / f"{prompt_type}.jsonl", "w") as f:
            for idx, task in enumerate(tqdm(prompts)):
                result = api_prompting(model, task[prompt_type])
                json.dump({
                    "idx":idx,
                    "model":model,
                    "task":task[prompt_type],
                    "type":prompt_type,
                    "result":result
                }, f)
                f.write("\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-path", default="./perturbation_prompt", type=Path)
    parser.add_argument("--output-path", default="./perturbed_text", type=Path)
    parser.add_argument("--dataset", default="qasper", choices=["qasper", "mbppplus", "cnn"], type=str)
    parser.add_argument("--model", default="gpt-4.1", type=str)
    args = parser.parse_args()
    main(args.prompt_path, args.output_path, args.dataset, args.model)