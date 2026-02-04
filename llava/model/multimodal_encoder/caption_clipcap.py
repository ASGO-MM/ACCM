import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from IPython.display import Image
import ipdb
import time


model_path = r'/public/home/renwu04/fmy/data/ClipCap/ClipCap_mlp_conceptual_weights.pt'
gpt_path = r'/public/home/renwu04/fmy/data/gpt2'
clip_path = r"/public/home/renwu04/fmy/data/ViT-B-32.pt"

img_path = r'/public/home/renwu04/fmy/LLaVA-PruMerge-text-3/images/llava_logo.png'

device_number = 1


N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')

CUDA = get_device


#@title Model

class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        ## ipdb.set_trace()
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, gptPath, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gptPath)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]    # 768
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self
    

#@title Caption prediction

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))   # text -> index
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)    # index -> embedding

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                ## ipdb.set_trace()
                try:
                    logits[:, indices_to_remove] = filter_value
                except RuntimeError as e:
                    print(e)
                    # ipdb.set_trace()
                
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)   # index -> embedding
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)    # index -> text
            generated_list.append(output_text)

    return generated_list[0]

def generate_nucleus(model, tokenizer, p: float = 0.9, prompt=None, embed=None,
                     entry_length=67, temperature=1., stop_token: str = '.'):
    """
    使用核采样生成文本。
    
    参数:
    - model: 自定义的 GPT-2 模型。
    - tokenizer: 与模型匹配的分词器。
    - p: 核采样的累积概率阈值（0 < p <= 1）。
    - prompt: 初始提示语。
    - embed: 直接使用的嵌入。
    - entry_length: 最大生成长度。
    - temperature: 控制生成的多样性。
    - stop_token: 停止生成的标记。

    返回:
    - 生成的文本。
    """
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    device = next(model.parameters()).device
    is_stopped = False  # 标记生成是否完成
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            probs = torch.softmax(logits, dim=-1)
            
            # 核采样：获取累积概率小于 p 的词
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            indices_to_remove = cumulative_probs > p
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = False

            sorted_probs[indices_to_remove] = 0
            sorted_probs /= sorted_probs.sum()  # 归一化

            # 根据处理后的概率分布进行采样
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(dim=-1, index=next_token)

            # 将新 token 添加到生成序列中
            tokens = next_token if tokens is None else torch.cat((tokens, next_token), dim=1)
            next_token_embed = model.gpt.transformer.wte(next_token.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)

            # 检查是否生成停止标记
            if next_token.item() == stop_token_index:
                is_stopped = True
                break

    # 解码生成的 tokens
    output_text = tokenizer.decode(tokens.squeeze().cpu().numpy(), skip_special_tokens=True)
    return output_text


def generate_greedy(model, tokenizer, prompt=None, embed=None,
                    entry_length=67, temperature=1., stop_token: str = '.'):
    """
    Greedy Search decoding function.
    
    Parameters:
        model: The language model.
        tokenizer: Tokenizer to encode and decode text.
        prompt: Initial text prompt.
        embed: Optional embeddings for initialization.
        entry_length: Max length of the generated sequence.
        temperature: Sampling temperature (lower values make the model deterministic).
        stop_token: Token that stops the generation.
        
    Returns:
        output_text: The generated text using Greedy Search.
    """
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    device = next(model.parameters()).device
    with torch.no_grad():
        # Initialize tokens and embeddings
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
            generated = model.gpt.transformer.wte(tokens)

        for i in range(entry_length):
            # Forward pass through the model
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            
            # Apply temperature and get probabilities
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            probabilities = logits.softmax(-1)
            
            # Select the token with the highest probability
            next_token = probabilities.argmax(dim=-1).unsqueeze(0)
            
            # Check for stop token
            if next_token.item() == stop_token_index:
                break

            # Append the new token to the sequence
            next_token_embed = model.gpt.transformer.wte(next_token)
            generated = torch.cat((generated, next_token_embed), dim=1)
            tokens = next_token if tokens is None else torch.cat((tokens, next_token), dim=1)

        # Decode the generated sequence
        output_list = tokens.squeeze().cpu().numpy()
        output_text = tokenizer.decode(output_list)
    return output_text


def generate_diverse_beam(model, tokenizer, beam_size: int = 6, group_size: int = 2, diversity_penalty: float = 1.0,
                          prompt=None, embed=None, entry_length=67, temperature=1., stop_token: str = '.'):
    """
    Diverse Beam Search decoding function.

    Parameters:
        model: The language model.
        tokenizer: Tokenizer to encode and decode text.
        beam_size: Total number of beams to maintain.
        group_size: Number of groups for diversity (beam_size must be divisible by group_size).
        diversity_penalty: Penalty for repeated tokens across groups.
        prompt: Initial text prompt.
        embed: Optional embeddings for initialization.
        entry_length: Max length of the generated sequence.
        temperature: Sampling temperature (lower values make the model deterministic).
        stop_token: Token that stops the generation.

    Returns:
        output_texts: List of generated texts ranked by their scores.
    """
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    device = next(model.parameters()).device
    group_count = beam_size // group_size  # Number of groups
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    tokens, scores = None, None

    with torch.no_grad():
        # Initialize tokens and embeddings
        if embed is not None:
            generated = embed
        else:
            tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
            generated = model.gpt.transformer.wte(tokens)

        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                # Initialize scores and tokens
                scores, next_tokens = logits.topk(beam_size, -1)
                # Expand for beam size
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                # Apply diversity penalty within groups
                for group in range(group_count):
                    group_start = group * group_size
                    group_end = group_start + group_size
                    group_logits = logits[group_start:group_end]
                    
                    # Penalize previously selected tokens in this group
                    for beam_idx in range(group_start, group_end):
                        for token in tokens[beam_idx]:
                            group_logits[:, token] -= diversity_penalty

                    logits[group_start:group_end] = group_logits

                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]

                # Select top candidates across all beams
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            # Update embeddings
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            ## ipdb.set_trace()
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

def generate_top_k(model, tokenizer, top_k: int = 5, prompt=None, embed=None,
                   entry_length=67, temperature=1., stop_token: str = '.'):
    """
    Top-k Sampling decoding function.
    
    Parameters:
        model: The language model.
        tokenizer: Tokenizer to encode and decode text.
        top_k: Number of top tokens to consider for sampling.
        prompt: Initial text prompt.
        embed: Optional embeddings for initialization.
        entry_length: Max length of the generated sequence.
        temperature: Sampling temperature (lower values make the model deterministic).
        stop_token: Token that stops the generation.
        
    Returns:
        output_text: The generated text using Top-k Sampling.
    """
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    device = next(model.parameters()).device
    with torch.no_grad():
        # Initialize tokens and embeddings
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
            generated = model.gpt.transformer.wte(tokens)

        for i in range(entry_length):
            # Forward pass through the model
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits   # torch.Size([1, 10, 50257])
            ## ipdb.set_trace()
            # Apply temperature scaling
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)   # torch.Size([1, 50257])
            probabilities = logits.softmax(-1)    # torch.Size([1, 50257])
            
            # Select top-k tokens and their probabilities
            top_k_probs, top_k_tokens = probabilities.topk(top_k, dim=-1)
            
            # Normalize the probabilities
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
            # Sample a token from the top-k
            sampled_token = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_tokens.gather(-1, sampled_token)
            
            # Check for stop token
            if next_token.item() == stop_token_index:
                break

            # Append the sampled token to the sequence
            next_token_embed = model.gpt.transformer.wte(next_token)
            generated = torch.cat((generated, next_token_embed), dim=1)
            tokens = next_token if tokens is None else torch.cat((tokens, next_token), dim=1)

        # Decode the generated sequence
        output_list = tokens.squeeze().cpu().numpy()
        output_text = tokenizer.decode(output_list)
    return output_text

# device = CUDA(device_number) 
# clip_model, preprocess = clip.load(clip_path, device=device, jit=False)
# tokenizer = GPT2Tokenizer.from_pretrained(gpt_path)

# prefix_length = 10
# model = ClipCaptionModel(gpt_path, prefix_length)
# ## ipdb.set_trace()
# model.load_state_dict(torch.load(model_path, map_location=CPU), strict = False) 

# model = model.eval() 
# model = model.to(device)

# #@title Inference
# use_beam_search = True #@param {type:"boolean"}  

# image = io.imread(img_path)
# pil_image = PIL.Image.fromarray(image)
# #pil_img = Image(filename=UPLOADED_FILE)
# #display(pil_image)

# image = preprocess(pil_image).unsqueeze(0).to(device)      # torch.Size([1, 3, 224, 224])
# ## ipdb.set_trace()
# start_time = time.time()
# with torch.no_grad():
#     # if type(model) is ClipCaptionE2E:
#     #     prefix_embed = model.forward_image(image)
#     # else:
#     prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)   # (1, 512)
#     ## ipdb.set_trace()
#     clip_time = time.time()
#     prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)   # torch.Size([1, 10, 768])
#     project_time = time.time()
# if use_beam_search:
#     generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
# else:
#     generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
# end_time = time.time()

# print('\n')
# print(generated_text_prefix)
# print(f"执行时间：{end_time - start_time}秒")
# print(f"clip 执行时间：{clip_time - start_time}秒")
# print(f"projector 执行时间：{project_time - clip_time}秒")
# print(f"gpt 执行时间：{end_time - project_time}秒")