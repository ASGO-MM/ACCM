import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection
from transformers import GPT2Tokenizer

import matplotlib.pyplot as plt
import numpy as np
import json
import ipdb
import math
from .clipcap_model import ClipCaptionModel
from .decoding_scheme import generate_beam, generate_nucleus, generate_diverse_beam, generate_top_k, generate_greedy
import clip
from .model import longclip
from .selector_transformer import TransformerSelector
from enum import Enum


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

def complement_idx(idx, dim):
    #ipdb.set_trace()
    ab = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        ab = ab.unsqueeze(0)
    ab = ab.expand(*dims)
    masked = torch.scatter(ab, -1, idx, 0)   # 把 ab 中 idx 位置的元素置为0   torch.Size([N, dim])
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))    # 
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()

    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((attn_np > upper_bound))[0]

    ratio = len(outlier_indices) / len(attn_np)
    return ratio


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer # default: -2
        #self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.select_feature = 'cls_patch'
        self.total_tokens = 0
        
        self.visual_token_num = 36
        self.text_num = 20

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        #self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)
        self.text_tower.requires_grad_(False)

        self.captionModel = ClipCaptionModel('/home/fmy/data/gpt2', prefix_length=20, prefix_size=768, clip_length=576-self.visual_token_num, num_layers=4, mapping_type=MappingType.Transformer)
        captionModel_path = '/home/fmy/data/ClipCap/checkpoints-end2end/coco_prefix-009.pt'
        print(captionModel_path)
        self.captionModel.load_state_dict(torch.load(captionModel_path, map_location=torch.device('cpu')))   # strict = False
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('/home/fmy/data/gpt2')
        self.captionModel.eval()
        self.captionModel.requires_grad_(False)
        #self.clipForCaption.requires_grad_(False)
        
        self.selector_mlp = TransformerSelector(embedding_dim=768, num_layers=4, num_heads=8, dim_feedforward=1536)
        selector_mlp_path = '/home/fmy/data/ClipCap/longclip_selector-transformer_one-word-20-epochs_5_bs4_4400/llava-RL_prefix-019.pt'
        print(selector_mlp_path)
        self.selector_mlp.load_state_dict(torch.load(selector_mlp_path, map_location=torch.device('cpu')))
        self.selector_mlp.eval()
        self.selector_mlp.requires_grad_(False)
        
        self.selector, _ = longclip.load("/home/fmy/data/longclip-L.pt", device=self.device)
        self.selector.eval()
        self.selector.requires_grad_(False)
        
        

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer] # penultimate layer output
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    
    
    def token_prune_merge_advanced_plus(self, images, prompt, if_adaptive=True, reduction_ratio = 1/8):
        '''
        version 24/03/2024 using the spacially sampled tokens to supplement the pruned tokens
        '''
        # token_indix_list = []
        # token_indix_dict = {}

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        cls_features = image_features[:, 0, :]
        image_features = image_features[:, 1:, :]
        assert image_features.shape[1] == 576
        B, N, C = image_features.shape

        select_num = self.visual_token_num    # 144, 36, 18, 288, 72
        max_num_text = self.text_num   # 70, 20, 12, 150, 40
        
        #ipdb.set_trace()
        if self.training:
            ipdb.set_trace()
            max_num_text = 80    # 60 for 104 tokens
            # max_num_text = 50   # for  pre-train  

        img_for_text = image_forward_outs.image_embeds
        img_for_text = img_for_text[:, 1:, :]    # torch.Size([B, 576, 768])
        img_for_text = img_for_text / img_for_text.norm(dim=2, keepdim=True)

        #ipdb.set_trace()
        assert len(prompt) == B

        # text 选 patch
        indice_list = []
        for index, question in enumerate(prompt):
            inputs_text = self.tokenizer(question, padding=True, return_tensors="pt").to(self.device)
            # 截断长度大于 77 的文本
            if(inputs_text['input_ids'].shape[-1] > 77):
                inputs_text['input_ids'] = inputs_text['input_ids'][:, :77]
                inputs_text['attention_mask'] = inputs_text['attention_mask'][:, :77]
                inputs_text['input_ids'][:, 76] = 49407
            
            outputs_text = self.text_tower(**inputs_text)
            text_embeds = outputs_text.text_embeds
            text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

            topk_num = int(max_num_text / len(question))
            if(len(question) > 16):    # 15
                topk_num = 5           # 4
            
            distances = (torch.sum(text_embeds ** 2, dim=1, keepdim=True)              # torch.Size([N, 576])   N: 问题个数
                     + torch.sum(img_for_text[index] ** 2, dim=1)
                     - 2 * torch.matmul(text_embeds, img_for_text[index].t()))

            _, indice = distances.topk(topk_num, dim=1)   # torch.Size([N, topk_num])

            if(len(question) > 16):    # 15
                indice = indice[:16, :]    # 15

            indice_list.append(indice.flatten())

        #ipdb.set_trace()

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]       # torch.Size([B, 577, 1024])
        desired_layer_q = outputs["desired_q"]       # torch.Size([B, 577, 1024])

        hook_handle_k.remove()
        hook_handle_q.remove()

        #ipdb.set_trace()
        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5      # torch.Size([B, 577, 577])
        attn = F.softmax(attn, dim=-1)    # torch.Size([B, 577, 577])

        cls_attn = attn[:, 0, 1:]    # torch.Size([B, 576])


        #ipdb.set_trace()
        # cls 选 patch
        if if_adaptive:
            idx_list = []
            for index_img in range(B):
                reduction_ratio = outlier_dectection(cls_attn[index_img])   #*3.5
                _, idx = torch.topk(cls_attn[index_img], int(N*reduction_ratio), dim=0, largest=True)  # [left_tokens] 
                idx_list.append(idx)

        #ipdb.set_trace()

        # 合并 text 和 cls 选择的 index
        new_idx_list = []
        for ind in range(B):
            new_indice = torch.cat((indice_list[ind], idx_list[ind]))
            #unique_indice = torch.unique(new_indice)
            unique_indice = torch.unique(new_indice, sorted=False)
            if(unique_indice.shape[0] > select_num):
                # unique_indice = unique_indice[ : select_num]
                unique_indice = unique_indice[-select_num : ]
            new_idx_list.append(torch.unique(unique_indice))

        idx_list = new_idx_list


        # uniform sample
        all_index = torch.arange(N).to(device=self.device)
        new_idx = torch.zeros((len(idx_list), select_num), dtype=torch.long).to(device=self.device)
        for i in range(len(idx_list)):
            need_num = select_num - idx_list[i].shape[0]
            if(need_num == 0):
                new_idx[i] = idx_list[i]
                continue
            step = math.ceil(N / need_num)
            arithmetic_sequence = torch.arange(0, 576, step).to(device=self.device)
            original_tensor_1d = idx_list[i].flatten().to(device=self.device)

            #filtered_sequence = arithmetic_sequence
            filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)            
            concatenated_tensor = torch.cat((original_tensor_1d, filtered_sequence), dim=0)
            if(concatenated_tensor.shape[0] < select_num):
                pad_num = select_num - concatenated_tensor.shape[0]
                mask = ~ torch.isin(all_index, concatenated_tensor)
                remain_index = all_index[mask]
                #pad_index = torch.random.choice(remain_index, size=(pad_num, ), replace=False)
                indices = torch.randperm(remain_index.size(0))[:pad_num]
                pad_index = remain_index[indices]
                concatenated_tensor = torch.cat((concatenated_tensor, pad_index), dim=0)
            
            assert concatenated_tensor.shape[0] == select_num
            new_idx[i] = concatenated_tensor

        idx = new_idx    # (B, 144)


        supplement_info = []
        proj_feature = image_forward_outs.image_embeds[:, 1:, :]   # (B, 576, 768)
        
        for i in range(B):
            all_idx = torch.arange(576).to(device=self.device)
            mask = ~torch.isin(all_idx, idx[i])
            remain_idx = all_idx[mask]
            remain_tokens = proj_feature[i][remain_idx].unsqueeze(dim=0)
            
        #   
            prefix_length = 20
            #ipdb.set_trace()
            prefix = remain_tokens
            prefix_embed = self.captionModel.clip_project(prefix).reshape(1, prefix_length, -1)
            
            # insert question embedding
            #ipdb.set_trace()
            assert len(prompt[i]) == 1
            prompt_id = torch.tensor(self.gpt_tokenizer.encode(prompt[i][0]))   # (N)
            prompt_id = prompt_id.unsqueeze(0).to(self.device)
            generated = self.captionModel.gpt.transformer.wte(prompt_id)   # torch.Size([1, N, 768])
            #generated = F.normalize(generated, dim=2)   # normalize question embedding
            prefix_embed = torch.cat((generated, prefix_embed), dim=1)   # q + v
            #
            caption_list = generate_beam(self.captionModel, self.gpt_tokenizer, embed=prefix_embed, beam_size=3)
            generated_text_prefix_1 = caption_list[0]
            generated_text_prefix_2 = caption_list[1]
            generated_text_prefix_3 = caption_list[2]

            
            prompt = prompt[i]
            if(len(prompt) > 12):
                prompt = prompt[ : 12]
            prefix = 'Question: '
            for question in prompt:
                question = question.strip()
                prefix = prefix + question
            prefix = prefix + '  Supplemental information for questions: '
            prefix = prefix.replace('\nAnswer with the option\'s letter from the given choices directly.', '')
            prefix = prefix.replace('\nAnswer the question using a single word or phrase.', '')
            tokens = []
            tokens.append(prefix + generated_text_prefix_1)
            tokens.append(prefix + generated_text_prefix_2)
            tokens.append(prefix + generated_text_prefix_3)
            #ipdb.set_trace()
            text = longclip.tokenize(tokens, truncate=True).to(self.device)
            with torch.no_grad():
                text_features = self.selector.encode_text(text)
            out = self.selector_mlp(text_features.unsqueeze(dim=0)).squeeze()
            indicator = torch.argmax(out)
            if indicator == 0:
                generated_text_prefix = generated_text_prefix_1
            if indicator == 1:
                generated_text_prefix = generated_text_prefix_2 
            if indicator == 2:
                generated_text_prefix = generated_text_prefix_3
            
            supplement_info.append(generated_text_prefix)
        #ipdb.set_trace()
        

        # original implementation
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]

        
        return cls_features, image_features, x_others, supplement_info

    

 
    @torch.no_grad()
    def forward(self, images, prompt):
        if type(images) is list:
            ipdb.set_trace()
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            cls_features, image_features, x_others, supplement_info = self.token_prune_merge_advanced_plus(images, prompt if_adaptive=True, reduction_ratio=1/8) 

        return cls_features, image_features, x_others, supplement_info

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
