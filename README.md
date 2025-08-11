# ACCM: Mitigating Information Loss under High Pruning Rates for Efficient Large Vision Language Models

Our work has been accepted by ACM Multimedia 2025. [[Paper](https://arxiv.org/pdf/2508.01236)]

<div align="center">
  <img src="https://github.com/ASGO-MM/ACCM/blob/main/images/Fig2_8.jpg" alt="Our approach" width="80%">
</div>

## Install
Prepare the environment as [LLaVA-1.5](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install).

## Benchmarks
We use 7 benchmarks, including MME, MMBench, POPE, MMVP, SEED, GQA and Flickr30k. For MMVP, we include it in our project. For Flickr30k, download from [our Huggingface homepage](https://huggingface.co/fumingyu064/ACCM). For preparing the other benchmarks, please refer to [LLaVA-1.5](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

## Evaluation
1. First download the checkpoints from [our Huggingface homepage](https://huggingface.co/fumingyu064/ACCM) and modify the path for the [caption model](https://github.com/ASGO-MM/ACCM/blob/0c5f93294fc2243791879b3aeda6dedde28a3ab7/llava/model/multimodal_encoder/clip_encoder.py#L97) and [classifier](https://github.com/ASGO-MM/ACCM/blob/0c5f93294fc2243791879b3aeda6dedde28a3ab7/llava/model/multimodal_encoder/clip_encoder.py#L106). The wieghts of [GPT2](https://huggingface.co/openai-community/gpt2) and [LongCLIP](https://huggingface.co/BeichenZhang/LongCLIP-L) are also needed and remember to modify the path in [clip_encoder.py](https://github.com/ASGO-MM/ACCM/blob/main/llava/model/multimodal_encoder/clip_encoder.py). 
2. Then set the visual token number in [clip_encoder.py](https://github.com/ASGO-MM/ACCM/blob/0c5f93294fc2243791879b3aeda6dedde28a3ab7/llava/model/multimodal_encoder/clip_encoder.py#L77) and [llava_arch.py](https://github.com/ASGO-MM/ACCM/blob/0c5f93294fc2243791879b3aeda6dedde28a3ab7/llava/model/llava_arch.py#L140).
3. Finally, run the scripts under [eval folder](https://github.com/ASGO-MM/ACCM/tree/main/scripts/v1_5/eval). Remember to modify the MODEL_PATH, CKPT_NAME and dataset path in the scripts.
For example, the evaluation for MME is:
```shell
bash scripts/v1_5/eval/mme.sh
```

## Training
Coming soon.
