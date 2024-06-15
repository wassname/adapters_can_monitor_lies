# 2024-06-09 16:05:45 

Started project using cookiecutter data science project template.

- https://github.com/huggingface/peft
  - we will use [quantization](https://colab.research.google.com/drive/1DOkD_5OUjFa0r5Ik3SgywJLJtEo2qLxO?usp=sharing)
    - https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
  - for peft+llama https://huggingface.co/blog/mlabonne/orpo-llama-3
  - for lightning and peft https://github.com/Lightning-AI/lit-llama/blob/main/finetune/adapter_v2.py
- for lie prompts:
  - https://github.com/wassname/LoRA_are_lie_detectors
- for collecting activations
  - https://github.com/wassname/uncensor_llms/blob/baukit/nbs/04_refusal_baukit.ipynb
- for lying llama
  - https://huggingface.co/failspy/Llama-3-8B-Instruct-abliterated
    - https://github.com/FailSpy/abliterator


# 2024-06-11 05:44:42

Hmm It's working in terms of probs. But coherency is lost. Maybe I need to generate more than one tokens?

oh I was taking the diff AFTER collection layers
could try using proper llama formatting too
can try gen

# 2024-06-13 06:09:59

So it's kind of worked a few times but it is
- unstable
- slow
- and I'm not doing the truthfull QA eval properly
- also my dataset is multishot lies.... but I need to measure how effective it is

TODO:
- improve dataset, sys prompts, eval
- for the retrain, collect hs over many samples? not sure if this would be a blunt instrument or good
- add training loss etc
- fix eval
- brainstorm other mechinterp composable options
  - erasing concepts
  - removing a concept while retaining a good one
  - steering


A dataset of prompts designed to elicit lies using system prompts and multi shot examples.  For a particular huggingface model, you can gent the subset of the dataset that the model can answer correctly, but doesn't.

First I need to make the dataset
- so system prompts often don't help... but can't hurt?
- first make a test dataset
- then input a hugginface model
- make it a seperate repo


#  2024-06-14 15:12:21

I moved ds creating to https://github.com/wassname/lie_elicitation_prompts
It's much better! I really check for knowledge, the code is cleaner etc

# 2024-06-15 13:30:14

I think I have most of the bugs out, it still a bit unstable, here's some ideas

- can I accumulate the original hidden_states.... but then they wouldn't be paired with the inputs?
- can I up the learning rate?
- right now I'm doing it over all hidden states... like in the paper, but maybe not?
  - OK we need to equally sample from a keep and retain dataset
- read the paper again with a highlighter
