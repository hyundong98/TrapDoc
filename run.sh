#!/bin/bash
# Commands for experiments on Qasper and gpt-4.1
python3 prepare_data.py
python3 generate_perturbation_prompt.py 
python3 llm_based_perturbation.py
python3 phantom_token_injection.py
python3 llm_inference.py
python3 evaluate.py
