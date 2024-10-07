import os
import re
import sys
import csv
import json
import time
import math
import torch
import numpy as np
import pandas as pd
import transformers
from collections import Counter
from bert_score import BERTScorer
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from nltk.translate.bleu_score import sentence_bleu
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        base_model: str = "../../LLM/ChatGLM3-6B",
        prompt_template: str = "alpaca_legacy",
        score: str = "abs_tfidf",
        instruction: str = "",
        # NW_text: str = "",
        temperature: float = 0.3,
        num_beams: int = 2,
        do_sample: bool = False,
        max_new_tokens: int = 1024,
        ratio_watermark_sentence: float = 1/3,
        read_bit: int = 8,
        map_way: bool = False,
        csv_file: str = "",
        model=None,
        tokenizer=None,
        rouge: str = "",
        bit_stream: str = "",
        bit_index: int = 0,
        zoom: int = 1,
        summarizer=None,
        key_strong=4096,
):

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)



    def evaluate(
            instruction,
            input=None,
            temperature=temperature,
            top_p=0.75,
            top_k=40,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            stream_output=False,
            **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)


        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output


        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        yield prompter.get_response(output)





    # ======================================================================================================================================================
    # =============================================== 1. Generate non-watermark text. (sample / beam search) ===============================================
    # ======================================================================================================================================================
    print("Generating non-watermark text ...")
    for non_watermark in evaluate(instruction):
        print("Instruction:\n", instruction)
        print("Non-watermark:\n", non_watermark)




    # ======================================================================================================================================================
    # ============================================================ 3. Generate watermark text. =============================================================
    # ======================================================================================================================================================
    print("Generating watermark text ...")
    input = None
    prompt = prompter.generate_prompt(instruction, input)


    def watermark_embedding(
            prompt,
            bit_stream,
            bit_index=0,
            top_k=10,
            max_new_tokens=2048,
            map_way=False,
            PRECISION=8,
            zoom=1,
    ):

        def bits2int(bits):
            res = 0
            for i, bit in enumerate(bits):
                res += bit * (2 ** i)
            return res


        def int2bits(inp, num_bits):
            if num_bits == 0:
                return []
            strlist = ('{0:0%db}' % num_bits).format(inp)
            return [int(strval) for strval in reversed(strlist)]


        def num_same_from_beg(bits1, bits2):
            assert len(bits1) == len(bits2)
            for i in range(len(bits1)):
                if bits1[i] != bits2[i]:
                    break
            return i


        watermark_bits = []
        # regenerate_prompt = ""
        # input = None

        with torch.no_grad():

            regenerate_inputs = tokenizer(prompt, return_tensors="pt")
            regenerate_input_ids = regenerate_inputs["input_ids"].to(device)
            watermark_bit = ''


            for i in range(max_new_tokens - 1):
                generation_output = model(regenerate_input_ids)
                log_prob = generation_output.logits
                prob = torch.softmax(log_prob, dim=-1)[:, -1, :].reshape(-1)
                prob = prob / prob.sum()
                prob, indices = prob.sort(descending=True)

                max_val = 2 ** PRECISION  # num of intervals
                cur_interval = [0, max_val]  # bottom inclusive, top exclusive
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range


                if prob[-1] < cur_threshold:
                    k = max(2, (prob < cur_threshold).nonzero()[0].item())
                    prob = prob[:k]
                    indices = indices[:k]

                prob = prob[:top_k]
                indices = indices[:top_k]

                if map_way:
                    prob = torch.pow(prob, zoom)

                prob = prob / prob.sum()
                prob = prob.double()
                prob *= cur_int_range
                prob = prob.round().long()

                cum_probs = prob.cumsum(0)
                overfill_index = (cum_probs > cur_int_range).nonzero()

                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]
                cum_probs += cur_int_range - cum_probs[-1]
                cum_probs += cur_interval[0]

                message_bits = bit_stream[bit_index: bit_index + PRECISION]
                message_bits = [int(_) for _ in message_bits]
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, PRECISION)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, PRECISION)))
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                prev = indices[selection].view(1, 1)

                gen = int(prev)
                regenerate_input_ids = torch.cat([
                    regenerate_input_ids,
                    torch.LongTensor([[gen]]).to(device)], dim=1
                ).to(device)

                watermark_bit += bit_stream[bit_index: bit_index + num_bits_encoded]
                bit_index += num_bits_encoded

                if 'LLaMA2' in base_model:
                    period = 2
                elif 'ChatGLM3' in base_model:
                    period = 2

                if gen == period:
                # if gen == period:
                    break

            output = tokenizer.decode(regenerate_input_ids[0], skip_special_tokens=True)
            output = output.split("Response:", 1)[1]

        watermark_bits.append(watermark_bit)
        return output, watermark_bits



    watermark_texts, watermark_bits = watermark_embedding(
        prompt,
        bit_stream=bit_stream,
        bit_index=bit_index,
        PRECISION=read_bit,
        map_way=map_way,
        zoom=zoom
    )


    print("Watermark:\n", watermark_texts)
    embed_watermark = ''.join(watermark_bits)
    print("Embedded watermark:", watermark_bits, len(embed_watermark))


    massage_certificate = {
        "P": instruction,
        "B": read_bit
    }
    json_string = json.dumps(massage_certificate)
    key = RSA.generate(key_strong)
    public_key = key.publickey()
    cipher_rsa = PKCS1_OAEP.new(public_key)
    encrypted_message = cipher_rsa.encrypt(json_string.encode())
    print(encrypted_message)





    # ======================================================================================================================================================
    # ===================================================================== 4. Extract. ====================================================================
    # ======================================================================================================================================================
    cipher_rsa = PKCS1_OAEP.new(key)
    decrypted_message = cipher_rsa.decrypt(encrypted_message)
    decrypted_dict = json.loads(decrypted_message.decode())
    prompt_value = decrypted_dict["P"]
    read_bit_value = decrypted_dict["B"]


    # prompt_value = "This is the prompt for verification failure."
    # read_bit_value = 4

    prompt = prompter.generate_prompt(prompt_value, input)

    def watermark_extraction(
            watermark_sens,
            top_k=10,
            prompt="",
            read_bits=4,
            map_way=False,
            zoom=1,
    ):
        def int2bits(inp, num_bits):
            strlist = ('{0:0%db}' % num_bits).format(inp)
            strlist = strlist[-num_bits:]
            return [int(strval) for strval in reversed(strlist)]

        def common_prefix(lst1, lst2):
            prefix = []
            for bit1, bit2 in zip(lst1, lst2):
                if bit1 == bit2:
                    prefix.append(str(bit1))
                else:
                    break
            return ''.join(prefix)



        with torch.no_grad():
            extracted_bitstream = []
            extracted_bit = ""
            regenerate_input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            watermark_sens_ids = tokenizer(watermark_sens, return_tensors="pt")["input_ids"].to(device)
            watermark_sens_ids = torch.cat([
                    watermark_sens_ids,
                    torch.LongTensor([[2]]).to(device)
                ], dim=1).to(device)
            if "LLaMA" in base_model:
                watermark_sens_ids = watermark_sens_ids[:, 2:]
            elif "ChatGLM" in base_model:
                watermark_sens_ids = watermark_sens_ids[:, 3:]

            i = 0
            # Watermark text to extract bits.
            while regenerate_input_ids[0][-1] != watermark_sens_ids[0][-1]:
                # print(i)
                generation_output = model(regenerate_input_ids)
                log_prob = generation_output.logits
                prob = torch.softmax(log_prob, dim=-1)[:, -1, :].reshape(-1)
                prob = prob / prob.sum()
                prob, indices = prob.sort(descending=True)

                max_val = 2 ** read_bits
                cur_interval = [0, max_val]
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range

                if prob[-1] < cur_threshold:
                    k = max(2, (prob < cur_threshold).nonzero()[0].item())
                    prob = prob[:k]
                    indices = indices[:k]

                prob = prob[:top_k]
                indices = indices[:top_k]

                if map_way:
                    prob = torch.pow(prob, zoom)

                prob = prob / prob.sum()
                prob = prob.double()
                prob *= cur_int_range
                prob = prob.round().long()

                cum_probs = prob.cumsum(0)
                overfill_index = (cum_probs > cur_int_range).nonzero()

                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]
                cum_probs += cur_int_range - cum_probs[-1]
                cum_probs += cur_interval[0]

                position_ids = watermark_sens_ids[0][i]

                if position_ids == watermark_sens_ids[0][-1]:
                    break
                else:
                    # print(len(watermark_idss), i, watermark_idss, position_ids, indices)
                    positio = torch.where(indices == position_ids)
                    # if len(positio[0]) > 0:
                    if len(positio[0]) == 0:
                        return extracted_bitstream
                    position = positio[0][0].item()

                    new_int_bottom = cum_probs[position - 1] if position > 0 else cur_interval[0]
                    new_int_top = cum_probs[position]
                    new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, read_bits)))
                    new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, read_bits)))
                    common_prefi = common_prefix(new_int_bottom_bits_inc, new_int_top_bits_inc)

                    regenerate_input_ids = torch.cat([
                        regenerate_input_ids,
                        torch.LongTensor([[position_ids]]).to(device)
                    ], dim=1).to(device)

                    extracted_bit += common_prefi
                    i += 1

            extracted_bitstream.append(extracted_bit)
        return extracted_bitstream

    start = time.time()
    extracted_bitstream = watermark_extraction(
        prompt=prompt,
        watermark_sens=watermark_texts,
        read_bits=read_bit_value,
        map_way=map_way,
        zoom=zoom,
    )
    end = time.time()
    Time_cost = end - start
    # print("Extracted watermark:", extracted_bitstream)
    extracted_watermark = ''.join(extracted_bitstream)





    # ======================================================================================================================================================
    # =========================================================== 5. Evaluate and save. ====================================================================
    # ======================================================================================================================================================
    print("Evaluating ...")


    def cosine_similarity(bitstream1, bitstream2):
        max_len = max(len(bitstream1), len(bitstream2))
        vector1 = np.array([int(bit) for bit in bitstream1.ljust(max_len, '0')])
        vector2 = np.array([int(bit) for bit in bitstream2.ljust(max_len, '0')])

        if not bitstream1 and not bitstream2:
            cosine_sim = 1.0
        else:
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            if norm1 == 0 or norm2 == 0:
                cosine_sim = 0
            else:
                cosine_sim = dot_product / (norm1 * norm2)
        return cosine_sim

    similarity = cosine_similarity(embed_watermark, extracted_watermark) * 100



    row_result = [
        instruction,
        non_watermark,
        watermark_texts,
        embed_watermark,
        len(embed_watermark),
        round(similarity, 2),
        Time_cost,
    ]
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(row_result)



# ================================================
# =================== 0. Main ====================
if __name__ == "__main__":

    # -------------- 0.1 Parameters --------------
    base_model = "../../LLM/LLaMA2-7B-chat"  # ChatGLM3-6B / LLaMA2-7B-chat
    load_8bit = False                         # True       / False

    max_tokens = 2048
    instuction_file = "IF_AlpacaFarm"         # IF_AlpacaFarm / QA_ELI5 / QA_FinQA
    generate_strategy = "sample"     # sample        / search
    score_way = "Entropy"            # Entropy       / abs_tfidf

    with open('bit_stream/bit_stream.txt', 'r', encoding='utf8') as f:
        bit_stream = f.read().strip()
        bit_stream += bit_stream

    # bit_stream = ''.join(format(ord(char), '08b') for char in "LLMspecial")
    # bit_index = 0

    ratio_watermark = 1     # The number of sentences that need to be embedded watermark
    read_bit = 16           # The number of bits read at one time
    map_way = False         # True / False
    zoom = 0.4              # 1.5 / 1.2 / 1.0 / 0.9 / 0.8 / 0.7 / 0.6 / 0.5 / 0.4
    rouge = "1"             # 2 / L

    if generate_strategy == "sample":
        temperature = 0.3
        do_sample = True
        num_beams = 1
    elif generate_strategy == "search":
        temperature = None
        do_sample = False
        num_beams = 4

    if not map_way:
        zoom = 1


    # -------------- 0.2 Load instruction and watermark-text files --------------
    Instruction_file = "./Instruction/" + instuction_file + ".txt"
    with open(Instruction_file, 'r', encoding='utf-8') as file:
        instruction = [line.strip() for line in file]

    csv_file = "./watermark_text/" + generate_strategy + "_" + instuction_file + "_" + str(ratio_watermark) + ".csv"



    # -------------- 0.3 Load model and tokenizer --------------
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        trust_remote_code=True,
    )

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map={"": device},
            low_cpu_mem_usage=True,
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

    if 'Llama-1' in base_model:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        print('Adding special tokens.')

        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(
                model.config.eos_token_id
            ),
            "bos_token": tokenizer.convert_ids_to_tokens(
                model.config.bos_token_id
            ),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1
                else tokenizer.pad_token_id
            ),
        })
    # if not load_8bit:
    #     model.bfloat16()  # seems to fix bugs for some users.


    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    summarizer = pipeline("summarization", model="../../LLM/BART-CNN")
    # -------------- 0.4 Run embedding watermark --------------
    print("Model:", base_model.split("/")[-1], " || ",
          "Sample:", do_sample, " || ",
          "Beam:", num_beams, " || ",
          "W-ratio:", ratio_watermark, " || ",
          "Score:", score_way, " || ",
          "Rouge-N:", rouge,
          )

    key_strong = 4096
    counter = 1

    if counter == 1:
        row = ["Instruction", "NW-text", "W-text", "Bits", "Num-Bits", "Success_rate", "Time (s)"]
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)


    for instr in instruction[counter - 1:]:
        print(f"================================== Position: {counter} ==================================")
        counter += 1
        bit_index = int(torch.randint(0, high=10000, size=(1,)))
        main(
            instruction=instr,
            # NW_text=NW_text,
            prompt_template="alpaca_legacy",
            base_model=base_model,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_tokens,
            ratio_watermark_sentence=ratio_watermark,
            read_bit=read_bit,
            map_way=map_way,
            csv_file=csv_file,
            model=model,
            tokenizer=tokenizer,
            score=score_way,
            rouge=rouge,
            bit_stream=bit_stream,
            bit_index=bit_index,
            zoom=zoom,
            summarizer=summarizer,
            key_strong=key_strong,
        )
