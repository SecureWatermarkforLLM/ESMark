import re
import torch
import math
import csv
import os
import numpy as np
import pandas as pd
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import difflib
from IPython.display import HTML
import webbrowser
import os

from utils.prompter import Prompter




def token_logits(text, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        return logits[0, :-1]



def compare_sequences(seq1, seq2):
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    modified = [False] * len(seq1)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'delete' or tag == 'insert':
            for i in range(i1, i2):
                if i < len(modified):
                    modified[i] = True

    return modified




if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

base_model: str = "../../LLM/LLaMA2-7B-chat"  # LLaMA2-7B-chat / ChatGLM3-6B
load_8bit: bool = False

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        trust_remote_code=True,
    )

if 'LLaMA3-8B' in base_model:
    print('Adding special tokens <unk=128002>.')
    model.config.unk_token_id = tokenizer.unk_token_id = 128002
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model.config.unk_token_id = tokenizer.unk_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

if 'LLaMA2-7B' in base_model:
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model.config.unk_token_id = tokenizer.unk_token_id
    model.config.pad_token_id = tokenizer.pad_token_id




folder_path = '../watermark_text/Ours/Ours_replace'
output_file = 'PPL.csv'
ppl_dict = {}
max_length = 0

prompter = Prompter("alpaca_legacy")
input = None
df_all = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith('7B_sample_substitution0.2.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        w_text_column = df['W-text'].tolist()
        mw_text_column = df['MW-text'].tolist()
        instruction_column = df['Instruction'].tolist()

        rank_all = []
        retained_rows = []
        tokens_scores = []


        for idx, mw_text in enumerate(mw_text_column):
            print(f"============ File: {filename} | Position: {idx} ==============")

            instruction = instruction_column[idx]
            w_text = w_text_column[idx]
            prompt = prompter.generate_prompt(instruction, input)
            combined_input = prompt + mw_text
            ocombined_input = prompt + w_text
            logits = token_logits(combined_input, model)
            # logits = token_logits(w_text, model)


            token_ids_all = tokenizer.encode(combined_input, return_tensors="pt")[0]
            otoken_ids_all = tokenizer.encode(ocombined_input, return_tensors="pt")[0]
            token_ids_prompt = tokenizer.encode(prompt, return_tensors="pt")[0]
            mw_text_ids = token_ids_all[len(token_ids_prompt):]
            w_text_ids = otoken_ids_all[len(token_ids_prompt):]


            # modified location
            mw_text_ids1 = mw_text_ids.tolist()
            w_text_ids1 = w_text_ids.tolist()
            modified = compare_sequences(w_text_ids1, mw_text_ids1)
            print("modified:", modified)


            rankings = []
            for token_idx, token_id in enumerate(mw_text_ids, start=len(token_ids_prompt)):
                logits_index = logits[token_idx - 1]
                sorted_indices = torch.argsort(logits[token_idx - 1], descending=True)
                rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1  # 1-based rank
                rankings.append(rank)


                if rank > 40:   score = 1
                elif rank > 20: score = 0.7
                elif rank > 10: score = 0.5
                elif rank > 5: score = 0.3
                else:  score = 0

                token_text = tokenizer.decode([token_id])
                tokens_scores.append((token_text, score))


            print(rankings)
            mw_text = tokenizer.decode(mw_text_ids)
            current_index = 0
            word_scores = []

            words = mw_text.split(' ')
            new_list = []
            for item in words:
                parts = item.split('\n')
                for index, part in enumerate(parts):
                    if part:
                        new_list.append(part)
                    if index < len(parts) - 1:
                        new_list.append('\n')


            for idx, word in enumerate(words):
                skip_tokens = 2 if idx == 0 else 1
                sub_tokens1 = tokenizer.encode(word)
                sub_tokens = [tokenizer.decode([token]) for token in sub_tokens1][skip_tokens:]
                max_score = 0
                for sub_token in sub_tokens:
                    while current_index < len(tokens_scores):
                        token, score = tokens_scores[current_index]
                        if token == sub_token:
                            max_score = max(max_score, score)
                            current_index += 1
                            break
                        current_index += 1
                word_scores.append((word, max_score))


            html_output = "<p>"
            for word, (word_text, score) in zip(mw_text, word_scores):
                # intensity = int(255 * score)
                # alpha = score
                # color = f"rgba(255, 255, {255 - intensity}, {alpha})"
                color = "rgba(255, 255, 255, 0)"
                if score == 1:
                    color = "rgb(64, 129, 177)"
                elif score == 0.7:
                    color = f"rgb(98, 155, 198)"
                elif score == 0.5:
                    intensity = int(255 * ((score - 0.5) / 0.2))
                    color = f"rgb(142, 183, 214)"
                elif score == 0.3:
                    intensity = int(255 * ((score - 0.3) / 0.2))
                    color = f"rgb(180, 207, 227)"
                elif score == 0:
                    color = "rgba(255, 255, 255, 0)"
                else:
                    color = "rgba(255, 255, 255, 0)"

                html_output += f"<span style='background-color:{color}'>{word_text}</span> "
            html_output += "</p><br><br>"

            HTML(html_output)

            with open("output.html", "a", encoding="utf-8") as file:
                file.write(html_output)
