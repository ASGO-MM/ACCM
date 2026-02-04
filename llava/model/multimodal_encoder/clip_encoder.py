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
from PIL import Image
from .decoding_scheme import generate_beam, generate_nucleus, generate_diverse_beam, generate_top_k, generate_greedy
import clip
from enum import Enum
from .reltr_models_adapt import build_model
import torchvision.transforms as T
from line_profiler import profile

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# VG classes
CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
            'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
            'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
            'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
            'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
            'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
            'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
            'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
            'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
            'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
            'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
            'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
            'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
            'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
            'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
            'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
            'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
            'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

class Config:
    def __init__(self, resume):
        self.lr_backbone = - 1e-5
        self.dataset = 'vg'
        self.batch_size = 1
        self.img_path = ''
        self.resume = resume
        self.device = 'cuda:0'
        self.backbone = 'resnet50'
        self.dilation = 'False'
        self.position_embedding = 'sine'
        self.enc_layers = 3    # 3
        self.dec_layers = 3    # 3
        self.dim_feedforward = 2048   # 2048
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.num_entities = 100
        self.num_triplets = 200
        self.pre_norm = False
        self.aux_loss = True
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.set_iou_threshold = 0.7
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.rel_loss_coef = 1
        self.eos_coef = 0.1
        self.return_interm_layers = False
        
        
    
    def __str__(self):
        """打印所有参数"""
        return "\n".join(f"{k}: {v}" for k, v in vars(self).items())



class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

def complement_idx(idx, dim):
    ## ipdb.set_trace()
    ab = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        ab = ab.unsqueeze(0)
    ab = ab.expand(*dims)
    masked = torch.scatter(ab, -1, idx, 0)   # torch.Size([N, dim])
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))    # 
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

# @profile
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
        self.select_feature = 'cls_patch'
        # getattr(args, 'mm_vision_select_feature', 'patch')
        self.total_tokens = 0
        reltr_ckpt = r'/home/cjb/data/reltr_ckpt/checkpoint0104.pth'
        self.args = Config(reltr_ckpt)
        
        print("adapt scene graph test.")
        print(reltr_ckpt)
        
        self.sample_index = 0
        
        self.triplet_feat_set = {}

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)
        self.text_tower.requires_grad_(False)
        
        self.sg_model, _, _ = build_model(self.args)
        ckpt = torch.load(self.args.resume)
        self.sg_model.load_state_dict(ckpt['model'], strict=False)
        self.sg_model.eval()

        print('load sg model.')

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
    
    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=self.device)
        return b

    
    @profile
    def token_prune_merge_advanced_plus(self, images, prompt, images_224, img_path, pil_image, if_adaptive=True, reduction_ratio = 1/8):
        '''
        version 24/03/2024 using the spacially sampled tokens to supplement the pruned tokens
        '''
        # token_indix_list = []
        # token_indix_dict = {}

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        cls_features = image_features[:, 0, :]
        image_features = image_features[:, 1:, :]
        assert image_features.shape[1] == 576
        B, N, C = image_features.shape

        select_num = 36 # 144, 72, 288, 576
        max_num_text = 20  # 70, 40, 150, 300
        
        if self.training:
            max_num_text = 80    # 60 for 104 tokens
            # max_num_text = 50   # for  pre-train  

        img_for_text = image_forward_outs.image_embeds
        img_for_text = img_for_text[:, 1:, :]    # torch.Size([B, 576, 768])
        img_for_text = img_for_text / img_for_text.norm(dim=2, keepdim=True)

        assert len(prompt) == B


        indice_list = []
        for index, question in enumerate(prompt):
            inputs_text = self.tokenizer(question, padding=True, return_tensors="pt").to(self.device)
            if(inputs_text['input_ids'].shape[-1] > 77):
                inputs_text['input_ids'] = inputs_text['input_ids'][:, :77]
                inputs_text['attention_mask'] = inputs_text['attention_mask'][:, :77]
                inputs_text['input_ids'][:, 76] = 49407
            
            outputs_text = self.text_tower(**inputs_text)
            text_embeds = outputs_text.text_embeds
            question_embeds = outputs_text.text_embeds 
            text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
            topk_num = int(max_num_text / len(question))
            if(len(question) > 16):    # 15
                topk_num = 5           # 4

            distances = (torch.sum(text_embeds ** 2, dim=1, keepdim=True)              # torch.Size([N, 576]) 
                     + torch.sum(img_for_text[index] ** 2, dim=1)
                     - 2 * torch.matmul(text_embeds, img_for_text[index].t()))

            _, indice = distances.topk(topk_num, dim=1)   # torch.Size([N, topk_num])

            if(len(question) > 16):    # 15
                indice = indice[:16, :]    # 15

            indice_list.append(indice.flatten())


        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]       # torch.Size([B, 577, 1024])
        desired_layer_q = outputs["desired_q"]       # torch.Size([B, 577, 1024])

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5      # torch.Size([B, 577, 577])
        attn = F.softmax(attn, dim=-1)    # torch.Size([B, 577, 577])

        cls_attn = attn[:, 0, 1:]    # torch.Size([B, 576])




        if if_adaptive:
            idx_list = []
            for index_img in range(B):
                reduction_ratio = outlier_dectection(cls_attn[index_img])   #*3.5
                _, idx = torch.topk(cls_attn[index_img], int(N*reduction_ratio), dim=0, largest=True)  # [left_tokens] 
                idx_list.append(idx)


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
        all_index = torch.arange(N, device=self.device)
        new_idx = torch.zeros((len(idx_list), select_num), dtype=torch.long, device=self.device)
        for i in range(len(idx_list)):
            need_num = select_num - idx_list[i].shape[0]
            if(need_num == 0):
                new_idx[i] = idx_list[i]
                continue
            step = math.ceil(N / need_num)
            arithmetic_sequence = torch.arange(0, 576, step, device=self.device)
            original_tensor_1d = idx_list[i].flatten()          # .to(device=self.device)

            #filtered_sequence = arithmetic_sequence
            filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d], device=self.device)            
            concatenated_tensor = torch.cat((original_tensor_1d, filtered_sequence), dim=0)
            if(concatenated_tensor.shape[0] < select_num):
                pad_num = select_num - concatenated_tensor.shape[0]
                mask = ~ torch.isin(all_index, concatenated_tensor)
                remain_index = all_index[mask]
                indices = torch.randperm(remain_index.size(0))[:pad_num]
                pad_index = remain_index[indices]
                concatenated_tensor = torch.cat((concatenated_tensor, pad_index), dim=0)
            
            assert concatenated_tensor.shape[0] == select_num
            new_idx[i] = concatenated_tensor

        idx = new_idx    # (B, 144)

        # original implementation
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
        
        try:
            if pil_image is None:
                im = Image.open(img_path[0]).convert('RGB')
                img = transform(im).unsqueeze(0)
        except RuntimeError as e:
            print(e)

        feat_for_sg = image_forward_outs.image_embeds
        feat_for_sg = feat_for_sg[:, 1:, :]    # torch.Size([B, 576, 768])
        try:
            sg_outputs = self.sg_model(pil_image, feat_for_sg, question_embeds)
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            ## ipdb.set_trace()

        # keep only predictions with 0.+ confidence
        probas = sg_outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = sg_outputs['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = sg_outputs['obj_logits'].softmax(-1)[0, :, :-1]
        # # ipdb.set_trace()

        keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                                probas_obj.max(-1).values > 0.3))
      
        sg_str = ''
      
        topk = 30
        
        keep_queries_temp = torch.arange(len(keep), device=keep.device)
        # keep_queries = keep_queries_temp[keep]
        keep_queries = torch.masked_select(keep_queries_temp, keep)

        indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
        keep_queries = keep_queries[indices]
        
        sub_list = []
        obj_list = []
        triplet = []
        
        
        for id_e in keep_queries:
            sub_now = CLASSES[probas_sub[id_e].argmax()]
            obj_now = CLASSES[probas_obj[id_e].argmax()]
            pred_now = REL_CLASSES[probas[id_e].argmax()]
            
            repeat = False
            for id_sub, sub in enumerate(sub_list):
                if sub_now == sub and obj_now == obj_list[id_sub]:
                    repeat = True
                    break
                
            if repeat == True:
                continue
            else:
                sub_list.append(sub_now)
                obj_list.append(obj_now)
                
                triplet_now = sub_now + ' ' + pred_now + ' ' + obj_now
                triplet.append(triplet_now)
        
        if triplet != []:
            chosen_triplet = []
            
            triplet_feat_list = []
            hit_triplet = []
            retained_triplet = []
            

            for ind_temp, triplet_temp in enumerate(triplet):
                if triplet_temp in self.triplet_feat_set.keys():

                    triplet_feat_list.append(self.triplet_feat_set[triplet_temp])
                    hit_triplet.append(triplet_temp)
                else:    

                    retained_triplet.append(triplet_temp)
            
            hit = False
            if hit_triplet != []:        
                triplet_feat = torch.stack(triplet_feat_list)
                hit = True
            
            retained = False
            if retained_triplet != []:

                triplet_text = self.tokenizer(retained_triplet, padding=True, return_tensors="pt").to(self.device)
                
                if(triplet_text['input_ids'].shape[-1] > 77):
                    triplet_text['input_ids'] = triplet_text['input_ids'][:, :77]
                    triplet_text['attention_mask'] = triplet_text['attention_mask'][:, :77]
                    triplet_text['input_ids'][:, 76] = 49407
                
                outputs_triplet = self.text_tower(**triplet_text)
                triplet_embeds = outputs_triplet.text_embeds     # torch.Size([n, 768])
                triplet_embeds = triplet_embeds / triplet_embeds.norm(dim=1, keepdim=True)     # torch.Size([n, 768])
                retained = True
                
                for ind_temp, triplet_temp in enumerate(retained_triplet):
                    self.triplet_feat_set[triplet_temp] = triplet_embeds[ind_temp]
            
            if hit and retained:
                triplet_new = hit_triplet + retained_triplet
                triplet_embeds = torch.cat((triplet_feat, triplet_embeds), dim=0)  # (n, 768)
            elif hit:
                triplet_new = hit_triplet
                triplet_embeds = triplet_feat
            elif retained:
                triplet_new = retained_triplet
                      
            topk_num = 7
            if topk_num > len(triplet_new):
                topk_num = len(triplet_new)
            
            '''
            distances = (torch.sum(triplet_embeds ** 2, dim=1, keepdim=True)              # torch.Size([n, 1])   n: triplet 个数
                        + torch.sum(text_embeds ** 2, dim=1)
                        - 2 * torch.matmul(triplet_embeds, text_embeds.t()))

            # # ipdb.set_trace()
            _, indice = distances.topk(topk_num, dim=0, largest=False)   # torch.Size([topk_num, 1])
            indice = indice.squeeze(dim=1)
            '''

            dot_products = torch.matmul(triplet_embeds, text_embeds.T)  
            _, indice = dot_products.topk(topk_num, dim=0, largest=True)   # torch.Size([topk_num, 1])
            indice = indice.squeeze(1)  
            
            for ind in indice:
                chosen_triplet.append(triplet_new[int(ind)])
            
        else:
            chosen_triplet = triplet
            
        for triplet_now in chosen_triplet:    
            if sg_str != '':
                sg_str += ', ' + triplet_now
            else:
                sg_str += triplet_now
        
        sg_str += '.'
        
        supplement_info = []
        supplement_info.append(sg_str)
        
        return cls_features, image_features, x_others, supplement_info

    

 
    @torch.no_grad()
    def forward(self, images, prompt, images_224, img_path, pil_image):
        if type(images) is list:
            ## ipdb.set_trace()
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            cls_features, image_features, x_others, supplement_info = self.token_prune_merge_advanced_plus(images, prompt, images_224, img_path, pil_image, if_adaptive=True, reduction_ratio=1/8) 

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
