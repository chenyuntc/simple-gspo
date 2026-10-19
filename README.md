# GSPO: Group Sequence Policy Optimization

A minimal implementation of GSPO from scratch for code generation tasks, based on the paper [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071).

Finetuning `Qwen/Qwen2.5-3B-Instruct` for *openai/human-eval*.

Results (holdout set: the last 16 HumanEval problems; training uses the other 148):

* Pass rate of base model: 3/16 (primarily formatting issues + code accuracy)

* Pass rate of finetuned model: 13/16

![result](./result.png)


Install dependencies:

```bash
pip install torch transformers datasets tqdm huggingface_hub
```

Run the code with (8x H200)

```bash
torchrun --nproc_per_node=8 main.py
```

Key references:
1. https://github.com/mingyin0312/RLFromScratch
2. https://arxiv.org/abs/2507.18071

