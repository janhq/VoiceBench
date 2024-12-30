# VoiceBench

This repo contains the code and data of:
[VoiceBench: Benchmarking LLM-Based Voice Assistants](https://arxiv.org/abs/2410.17196)

## News
* **`2024.12.11`** Updated the VoiceBench Leaderboard to include `mmsu`.
* **`2024.12.10`** Added a curated list of awesome voice assistants.
* **`2024.11.24`** Expanded the test samples in VoiceBench to include `mmsu`, covering 12 diverse domains from `mmlu-pro`.
* **`2024.11.12`** Updated the VoiceBench Leaderboard to include: 1) Mini-Omni2, GPT-4o-Audio, and Whisper-v3+GPT-4o, and 2) multiple-choice QA from OpenBookQA.
* **`2024.10.30`** Expanded the test samples in VoiceBench to include: 1) the complete set of open-ended QA from `alpacaeval`, and 2) multiple-choice QA from `openbookqa`.

## Table of Contents
- [**Leaderboard**](#leaderboard)
- [**Setup**](#setup)
- [**Dataset**](#dataset)
- [**Evaluation**](#evaluation)
- [**Awesome Voice Assistants**](#awesome-voice-assistants)
- [**Citation**](#citation)

## Leaderboard

| Rank | Model                         | AlpacaEval | CommonEval | SD-QA | MMSU  | OpenBookQA | IFEval | AdvBench | Overall |
|:----:|-------------------------------|:----------:|:----------:|:-----:|:-----:|:----------:|:------:|:--------:|:-------:|
|  1   | Whisper-v3-large+GPT-4o       |    4.80    |    4.47    | 75.77 | 81.69 |   92.97    | 76.51  |  98.27   |  87.23  |
|  2   | GPT-4o-Audio                  |    4.78    |    4.49    | 75.50 | 80.25 |   89.23    | 76.02  |  98.65   |  86.42  |
|  3   | Whisper-v3-large+LLaMA-3.1-8B |    4.53    |    4.04    | 70.43 | 62.43 |   81.54    | 69.53  |  98.08   |  79.06  |
|  4   | Whisper-v3-turbo+LLaMA-3.1-8B |    4.55    |    4.02    | 58.23 | 62.04 |   72.09    | 71.12  |  98.46   |  76.16  |
|  5   | Ultravox-v0.4.1-LLaMA-3.1-8B  |    4.55    |    3.90    | 53.35 | 47.17 |   65.27    | 66.88  |  98.46   |  71.45  |
|  6   | Whisper-v3-turbo+LLaMA-3.2-3B |    4.45    |    3.82    | 49.28 | 51.37 |   60.66    | 69.71  |  98.08   |  70.66  |
|  7   | MERaLiON                      |    4.50    |    3.77    | 55.06 | 34.95 |   27.23    | 62.93  |  94.81   |  62.91  |
|  8   | GLM-4-Voice                   |    3.97    |    3.42    | 36.98 | 39.75 |   53.41    | 25.92  |  88.08   |  55.99  |
|  9   | DiVA                          |    3.67    |    3.54    | 57.05 | 25.76 |   25.49    | 39.15  |  98.27   |  55.70  |
|  10  | Qwen2-Audio                   |    3.74    |    3.43    | 35.71 | 35.72 |   49.45    | 26.33  |  96.73   |  55.35  |
|  11  | KE-Omni-v1.5                  |    3.82    |    3.20    | 31.20 | 32.27 |   58.46    | 15.00  |  100.00  |  53.90  |
|  12  | Megrez-3B-Omni                |    3.50    |    2.95    | 25.95 | 27.03 |   28.35    | 25.71  |  87.69   |  46.25  |
|  13  | Ichigo                        |    3.79    |    3.17    | 36.53 | 25.63 |   26.59    | 21.59  |  57.50   |  43.86  |
|  14  | LLaMA-Omni                    |    3.70    |    3.46    | 39.69 | 25.93 |   27.47    | 14.87  |  11.35   |  37.51  |
|  15  | VITA                          |    3.38    |    2.15    | 27.94 | 25.70 |   29.01    | 22.82  |  26.73   |  34.68  |
|  16  | Mini-Omni2                    |    2.32    |    2.18    | 9.31  | 24.27 |   26.59    | 11.56  |  57.50   |  31.32  |
|  17  | Mini-Omni                     |    1.95    |    2.02    | 13.92 | 24.69 |   26.59    | 13.58  |  37.12   |  27.90  |
|  18  | Moshi                         |    2.01    |    1.60    | 15.64 | 24.04 |   25.93    | 10.12  |  44.23   |  27.47  |


We encourage you to submit new voice assistant results directly through the issue tracker. The ranking list will be updated accordingly.

## Setup
```shell
conda create -n voicebench python=3.10
conda activate voicebench
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23 --no-deps
pip install -r requirements.txt
```

## Dataset

The data used in this project is available at [VoiceBench Dataset](https://huggingface.co/datasets/hlt-lab/voicebench) hosted on Hugging Face.

You can access it directly via the link and integrate it into your project by using the Hugging Face `datasets` library.

### How to Use the Dataset

To load the dataset in your Python environment:

```python
from datasets import load_dataset

# Load the VoiceBench dataset
# Available subset: alpacaeval, commoneval, sd-qa, ifeval, advbench, ...
dataset = load_dataset("hlt-lab/voicebench", 'alpacaeval')
```

### Available Data

| Subset          | # Samples | Audio Source |       Task Type       |
|-----------------|:---------:|:------------:|:---------------------:|
| alpacaeval      |    199    |  Google TTS  |     Open-Ended QA     |
| alpacaeval_full |    636    |  Google TTS  |     Open-Ended QA     |
| commoneval      |    200    |    Human     |     Open-Ended QA     |
| openbookqa      |    455    |  Google TTS  |  Multiple-Choice QA   |
| mmsu            |    3,074  |  Google TTS  |  Multiple-Choice QA   |
| sd-qa           |    553    |    Human     |  Reference-Based QA   |
| mtbench         |    46     |  Google TTS  |     Multi-Turn QA     |
| ifeval          |    345    |  Google TTS  | Instruction Following |
| advbench        |    520    |  Google TTS  |        Safety         |


**PS**: `alpacaeval` contains `helpful_base` and `vicuna` data, while `alpacaeval_full` is constructed with the complete data. `alpacaeval_full` is used in the leaderboard.


## Evaluation
### Step 1: Get the Voice Assistant's Response
To obtain the responses from the voice assistant model, run the following command:
```shell
python main.py --model naive --data alpacaeval --split test --modality audio
```

**Supported Arguments:**
- `--model`: Specifies the model to use for generating responses. Replace `naive` with the model you want to test (e.g., `qwen2`, `diva`).
- `--data`: Selects the subset of the dataset. Replace `alpacaeval` with other subsets like `commoneval`, `sd-qa`, etc., depending on your evaluation needs.
- `--split`: Chooses the data split to evaluate.
    - For most datasets (`alpacaeval`, `commoneval`, `ifeval`, `advbench`), use `test` as the value.
    - For the `sd-qa` subset, you should provide a region code instead of `test`, such as `aus` for Australia, `usa` for the United States, etc.
- `--modality`: Use `audio` for spoken instructions, `text` for text-based instructions.

This will generate the output and save it to a file named naive-alpacaeval-test-audio.jsonl.

### Step2: Automatic GPT-4 Evaluation
For datasets like `alpacaeval`, `commoneval`, and `sd-qa`, we use `gpt-4o-mini` to evaluate the responses. Run the following command to get the GPT score:
```shell
python api_judge.py --src_file naive-alpacaeval-test-audio.jsonl
```
The GPT evaluation scores will be saved to `result-naive-alpacaeval-test-audio.jsonl`.

**Note:** This step should be skipped for the `advbench` and `ifeval` subsets, as they are not evaluated using GPT-4.

### Step3: Get the Final Results
To generate the final evaluation results, run:
```shell
python evaluate.py --src_file result-naive-alpacaeval-test-audio.jsonl --evaluator open
```
**Supported Arguments:**
- `--evaluator`: Specifies the evaluator type:
    - Use `open` for `alpacaeval` and `commoneval`.
    - Use `qa` for `sd-qa`.
    - Use `ifeval` for `ifeval`.
    - Use `harm` for `advbench`.
    - Use `mcq` for `openbookqa`.

## Awesome Voice Assistants
| Title                                                                                                                                                                                                                                                |    Date    |                        Code                        |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------:|:--------------------------------------------------:|
| [**SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training**](https://arxiv.org/abs/2412.15649)                                                                                                                           | 2024-12-20 |                         --                         |
| [**Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition**](https://arxiv.org/abs/2412.09501v1) &nbsp; ![Star](https://img.shields.io/github/stars/dvlab-research/Lyra.svg?style=social&label=Star)                                     | 2024-12-12 |  [Github](https://github.com/dvlab-research/Lyra)  |
| [**GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot**](https://arxiv.org/abs/2412.02612) &nbsp; ![Star](https://img.shields.io/github/stars/THUDM/GLM-4-Voice.svg?style=social&label=Star)                                  | 2024-12-03 |   [Github](https://github.com/THUDM/GLM-4-Voice)   |
| [**Advancing Speech Language Models by Scaling Supervised Fine-Tuning with Over 60,000 Hours of Synthetic Speech Dialogue Data**](https://arxiv.org/abs/2412.01078)                                                                                  | 2024-12-02 |                         --                         |
| [**SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation**](https://arxiv.org/abs/2411.18138)                                                                                                                           | 2024-11-27 |                         --                         |
| [**Ultravox: An Open-Weight Alternative to GPT-4o Realtime**](https://www.ultravox.ai/blog/ultravox-an-open-weight-alternative-to-gpt-4o-realtime) &nbsp; ![Star](https://img.shields.io/github/stars/fixie-ai/ultravox.svg?style=social&label=Star) | 2024-11-12 |   [Github](https://github.com/fixie-ai/ultravox)   |
| [**Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM**](https://arxiv.org/abs/2411.00774) &nbsp; ![Star](https://img.shields.io/github/stars/VITA-MLLM/Freeze-Omni.svg?style=social&label=Star)                   | 2024-11-01 | [Github](https://github.com/VITA-MLLM/Freeze-Omni) |
| [**OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation**](https://arxiv.org/abs/2410.17799)                                                                                                                                         | 2024-10-23 |                         --                         |
| [**Ichigo: Mixed-Modal Early-Fusion Realtime Voice Assistant**](https://arxiv.org/abs/2410.15316)       &nbsp; ![Star](https://img.shields.io/github/stars/janhq/ichigo.svg?style=social&label=Star)                                                 | 2024-10-20 |     [Github](https://github.com/janhq/ichigo)      |
| [**Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities**](https://arxiv.org/abs/2410.11190) &nbsp; ![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni2.svg?style=social&label=Star)                       | 2024-10-15 |  [Github](https://github.com/gpt-omni/mini-omni2)  |
| [**Ocean-omni: To Understand the World with Omni-modality**](https://arxiv.org/abs/2410.08565v3)                                                                                                                                                     | 2024-10-11 |                         --                         |
| [**Distilling an End-to-End Voice Assistant Without Instruction Training Data**](https://arxiv.org/abs/2410.02678)                                                                                                                                   | 2024-10-03 |                         --                         |
| [**EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions**](https://arxiv.org/abs/2409.18042)                                                                                                                                 | 2024-09-26 |                         --                         |
| [**Moshi: a Speech-Text Foundation Model for Real-Time Dialogue**](https://arxiv.org/abs/2410.00037) &nbsp; ![Star](https://img.shields.io/github/stars/kyutai-labs/moshi.svg?style=social&label=Star)                                               | 2024-09-17 |   [Github](https://github.com/kyutai-labs/moshi)   |
| [**LLaMA-Omni: Seamless Speech Interaction with Large Language Models**](https://arxiv.org/abs/2409.06666) &nbsp; ![Star](https://img.shields.io/github/stars/ictnlp/LLaMA-Omni.svg?style=social&label=Star)                                         | 2024-09-10 |   [Github](https://github.com/ictnlp/LLaMA-Omni)   |
| [**Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming**](https://arxiv.org/abs/2408.16725) &nbsp; ![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni.svg?style=social&label=Star)                                     | 2024-08-29 |  [Github](https://github.com/gpt-omni/mini-omni)   |
| [**VITA: Towards Open-Source Interactive Omni Multimodal LLM**](https://arxiv.org/abs/2408.05211) &nbsp; ![Star](https://img.shields.io/github/stars/VITA-MLLM/VITA.svg?style=social&label=Star)                                                     | 2024-08-09 |    [Github](https://github.com/VITA-MLLM/VITA)     |
| [**Qwen2-Audio Technical Report**](https://arxiv.org/abs/2407.10759) &nbsp; ![Star](https://img.shields.io/github/stars/QwenLM/Qwen2-Audio.svg?style=social&label=Star)                                                                              | 2024-07-15 |  [Github](https://github.com/QwenLM/Qwen2-Audio)   |
| [**PSLM: Parallel Generation of Text and Speech with LLMs for Low-Latency Spoken Dialogue Systems**](https://arxiv.org/abs/2406.12428)                                                                                                               | 2024-06-18 |                         --                         |
| [**LLaSM: Large Language and Speech Model**](https://arxiv.org/abs/2308.15930) &nbsp; ![Star](https://img.shields.io/github/stars/LinkSoul-AI/LLaSM.svg?style=social&label=Star)                                                                     | 2023-08-30 |   [Github](https://github.com/LinkSoul-AI/LLaSM)   |


## Citation
If you use the VoiceBench in your research, please cite the following paper:
```
@article{chen2024voicebench,
  title={VoiceBench: Benchmarking LLM-Based Voice Assistants},
  author={Chen, Yiming and Yue, Xianghu and Zhang, Chen and Gao, Xiaoxue and Tan, Robby T. and Li, Haizhou},
  journal={arXiv preprint arXiv:2410.17196},
  year={2024}
}
```
