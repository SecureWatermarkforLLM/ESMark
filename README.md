# An Entirely Extracted and Secure Multi-bit Watermark for Large Language Models.
<div style="text-align: center;">
<img src="./Figures/fig1.png" alt="1" title="1" width="800" height="250">
</div>


## 1. Datasets
Commonly used datasets in the field of natural language processing and watermarking are used to evaluate the performance of ESMark, which are the instruction following task: [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm), the question answering tasks: [ELI5](https://github.com/facebookresearch/ELI5) and [FinQA](https://sites.google.com/view/fiqa/home). You can get these datasets from [here](https://github.com/THU-KEG/WaterBench/tree/main/data/WaterBench).


## 2. Models
The models we used are LLama2-7B-chat / ChatGLM3-6B. You can get the LLama2-7B-chat from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and ChatGLM3-6B from [here](https://huggingface.co/THUDM/chatglm3-6b).


## 3. Conda Environment
Python 3.10

`pip install -r requirements.txt`


## 4. Multiple Scenarios
We designed full embedding and partial embedding scenarios, and the codes are run: `ESMark_full.py` and `ESMark_partial.py`.

<div style="text-align: center;">
<img src="./Figures/fig2.png" alt="1" title="1" width="600" height="500">
</div>
