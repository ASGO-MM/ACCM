import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import ipdb
import clip
import skimage.io as io
from PIL import Image
from clipcap_model import ClipCaptionModel, ClipCaptionPrefix
from clip_encoder import CLIPVisionTower
from llava_llama import LlavaLlamaForCausalLM
from llava_trainer import LengthGroupedSampler
from llava_datasets import *
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from decoding_scheme import generate_greedy, generate_nucleus, generate_top_k
#from selector import Selector
#from selector_mlp import Selector_Mlp
# from selector_transformer import TransformerSelector
# from model import longclip
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]     # 长度对齐
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))  # 补 -1
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0    # 小于0的置为0，大于等于0的不变
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        filename = os.path.join(self.img_dir, self.image_ids[item])
        image = io.imread(filename)
        image_tensor = self.preprocess(Image.fromarray(image))
        # prefix = self.prefixes[self.caption2embedding[item]]
        # assert self.caption2embedding[item] == item
        # if self.normalize_prefix:
        #     prefix = prefix.float()
        #     prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, image_tensor

    def __init__(self, data_path: str,  img_dir: str, prefix_length: int, gpt2_path: str, preprocess,
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.img_dir = img_dir
        self.preprocess = preprocess
        with open(data_path, 'r') as f:
            all_data = json.load(f)
        print("Data size is %0d" % len(all_data))
        sys.stdout.flush()
        # self.prefixes = all_data["clip_embedding"]
        # captions_raw = all_data["captions"]
        self.image_ids = [caption["image"] for caption in all_data]
        self.captions = [caption['caption'] for caption in all_data]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            ipdb.set_trace()
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []   # input_ids of caption
            #self.caption2embedding = []
            max_seq_len = 0
            for caption in self.captions:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
                #self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            #ipdb.set_trace()
            # self.max_seq_len = max_seq_len
            #with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
            #    pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))




def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def to_one_hot(tensor):
    """
    将输入张量转换为 one-hot 标签。
    
    Args:
        tensor (torch.Tensor): 输入一维张量。
        
    Returns:
        torch.Tensor: one-hot 标签张量。
    """

    assert len(tensor.shape) == 1
    # 找到最大值的索引
    max_index = torch.argmax(tensor, dim=0)
    
    # 创建一个全零张量
    one_hot = torch.zeros_like(tensor, dtype=torch.int32)
    
    # 将最大值对应位置置为 1
    one_hot[max_index] = 1
    
    return one_hot


def train(train_dataloader, model: ClipCaptionModel, vision_tower, llava_tokenizer, gpt2_tokenizer, args,
          lr: float = 1e-5, warmup_steps: int = 200, output_dir: str = ".", output_prefix: str = "", device=torch.device('cuda:0'), device_llava=torch.device('cuda:1')):

    #batch_size = args.bs
    print('lr: ', lr)
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    
    for k, v in model.named_parameters():
        print(k, v.requires_grad)
    
    vision_tower = vision_tower.to(device, dtype=torch.bfloat16)
    vision_tower.eval()
    vision_tower.requires_grad_(False)
    
    # llava_model = llava_model.to(dtype=torch.bfloat16)
    # llava_model = llava_model.to(device_llava)
    # llava_model.eval()
    # llava_model.requires_grad_(False)
    #ipdb.set_trace()
    
    # selector = selector.to(device)
    # selector.eval()
    # selector.requires_grad_(False)

    # selector_mlp = selector_mlp.to(device)
    # selector_mlp.train()

    
    # for k, v in selector.named_parameters():
    #     print(k, v.requires_grad)
    # for k, v in selector_mlp.named_parameters():
    #     print(k, v.requires_grad)
    
    
    #optimizer = AdamW(selector_mlp.parameters(), lr=lr)     # modified !!!
    ## train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    # )
    # save_config(args)

    #loss_list = []
    json_path = '/data16tb/fmy/ClipCap/llava_one-answer_with-cap.json'
    #criterion_nll = nn.NLLLoss()
    # if not os.path.exists(loss_curve_path):
    #     os.makedirs(loss_curve_path)
    for epoch in range(epochs):
        #skip_epoch = 0
        num_train = 0
        data_list = []
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, inputs in enumerate(train_dataloader):
            #ipdb.set_trace()
            #selector_mlp.zero_grad()     # modified !!!
            B = inputs['images'].shape[0]
            _, image_features, x_others, supplement_token = vision_tower(inputs['images'].to(dtype=torch.bfloat16), inputs['prompt'], inputs['images_224'].to(dtype=torch.bfloat16))
            #ipdb.set_trace()
            generated_text_nucleus = []
            generated_text_topk = []
            generated_text_greedy = []
            for i in range(B):
                prefix = supplement_token[i].unsqueeze(dim=0).to(dtype=torch.float32)
                model.eval()
                with torch.no_grad():
                    prefix_embed = model.clip_project(prefix).reshape(1, model.prefix_length, -1) 
                generated_text_nucleus.append(generate_nucleus(model, gpt2_tokenizer, embed=prefix_embed, p=0.9))
                generated_text_topk.append(generate_top_k(model, gpt2_tokenizer, embed=prefix_embed, top_k=10))
                generated_text_greedy.append(generate_greedy(model, gpt2_tokenizer, embed=prefix_embed))
            '''
            # ref probs
            inputs['x_others'] = image_features.to(device_llava)
            inputs['generated_text'] = []
            inputs['decoding_scheme'] = ''
            with torch.no_grad():
                ref_probs = llava_model(**inputs)['ans_probs']    # []
            # nucleus cap
            inputs['x_others'] = x_others.to(device_llava)  # (B, ratio, C)
            inputs['generated_text'] = generated_text_nucleus  # []  len=B
            inputs['decoding_scheme'] = 'nucleus'
            with torch.no_grad():
                nucleus_probs = llava_model(**inputs)['ans_probs'] 
            # topk cap
            inputs['x_others'] = x_others.to(device_llava)  # (B, ratio, C)
            inputs['generated_text'] = generated_text_topk  # []  len=B
            inputs['decoding_scheme'] = 'topk'
            with torch.no_grad():
                topk_probs = llava_model(**inputs)['ans_probs'] 
            # greedy cap
            inputs['x_others'] = x_others.to(device_llava)  # (B, ratio, C)
            inputs['generated_text'] = generated_text_greedy  # []  len=B
            inputs['decoding_scheme'] = 'greedy'
            with torch.no_grad():
                greedy_probs = llava_model(**inputs)['ans_probs'] 
                
            # process exception
            error_flag = False
            for i in range(B):
                if(nucleus_probs[i].shape != ref_probs[i].shape):
                    error_flag = True
                if(topk_probs[i].shape != ref_probs[i].shape):
                    error_flag = True
                if(greedy_probs[i].shape != ref_probs[i].shape):
                    error_flag = True
            if(error_flag == True):
                print()
                print("Shape error!")
                print(inputs['id'])
                for i in range(B):
                    print(ref_probs[i].shape, nucleus_probs[i].shape, topk_probs[i].shape, greedy_probs[i].shape)
                print()
                continue
            

            for i in range(B):
                
                P_nucleus = nucleus_probs[i].softmax(dim=-1)   # torch.Size([2, 32000])
                P_topk = topk_probs[i].softmax(dim=-1)
                P_greedy = greedy_probs[i].softmax(dim=-1)
                nucleus_id = P_nucleus.argmax(dim=1)    # torch.Size([2])
                topk_id = P_topk.argmax(dim=1)
                greedy_id = P_greedy.argmax(dim=1)
                correct_ans_id = llava_tokenizer(inputs['ori_data'][i]['conversations'][1]['value'], return_tensors="pt", padding="longest", max_length=llava_tokenizer.model_max_length, truncation=True).input_ids.squeeze().to(device_llava)   # torch.Size([2])
                
                try:
                    temp_nucleus = (nucleus_id[:-1] == correct_ans_id[1:])
                    temp_topk = (topk_id[:-1] == correct_ans_id[1:])
                    temp_greedy = (greedy_id[:-1] == correct_ans_id[1:])
                except IndexError as e:
                    print(e)
                    ipdb.set_trace()
                    continue
                except Exception as e:
                    print(e)
                    ipdb.set_trace()
                    continue
                
                if(False in temp_nucleus):
                    flag_nucleus = 0
                else:
                    flag_nucleus = 1
                if(False in temp_topk):
                    flag_topk = 0
                else:
                    flag_topk = 1
                if(False in temp_greedy):
                    flag_greedy = 0
                else:
                    flag_greedy = 1
                flag_all = flag_nucleus + flag_topk + flag_greedy
                #ipdb.set_trace()
                if(flag_all < 3 and flag_all > 0):
                    new_data = inputs['ori_data'][i]
                    new_data['nucleus_text'] = generated_text_nucleus[i]
                    new_data['topk_text'] = generated_text_topk[i]
                    new_data['greedy_text'] = generated_text_greedy[i]
                    new_data['nucleus_flag'] = flag_nucleus
                    new_data['topk_flag'] = flag_topk
                    new_data['greedy_flag'] = flag_greedy
                    data_list.append(new_data)
                    num_train += 1
                #ipdb.set_trace()
            '''
            
            for i in range (B):
                new_data = inputs['ori_data'][i]
                new_data['nucleus_text'] = generated_text_nucleus[i]
                new_data['topk_text'] = generated_text_topk[i]
                new_data['greedy_text'] = generated_text_greedy[i]
                data_list.append(new_data)
                num_train += 1
                #ipdb.set_trace()
            
            progress.set_postfix({"num_train": num_train})
            #progress.set_postfix({"loss_weight": loss_weight})
            progress.update()
            if (idx + 1) % 10000 == 0:
                f_out = open(json_path, 'w')
                json.dump(data_list, f_out)
            '''
            if (idx + 1) % 10000 == 0:
                torch.save(selector_mlp.state_dict(), os.path.join(output_dir, f"{output_prefix}_latest_{epoch+1}_{idx + 1}.pt"),)   # modified !!!
            if (idx + 1) % 500 == 0:
                plt.plot(loss_list)
                plt.xlabel("iterations")
                plt.ylabel("loss")
                plt.savefig(os.path.join(loss_curve_path, f"{output_prefix}_latest_{epoch+1}_{idx + 1}_iter.jpg"))
            '''    
        progress.close()
        f_out = open(json_path, 'w')
        json.dump(data_list, f_out)
        '''
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(selector_mlp.state_dict(), os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),)   # modified !!!
            plt.plot(loss_list)
            plt.xlabel("iterations")
            plt.ylabel("loss")
            plt.savefig(os.path.join(loss_curve_path, f"{output_prefix}_latest_{epoch + 1}_epoch.jpg"))
        '''
    return 1


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    # parser.add_argument('--img_dir', default='./checkpoints')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--gpt_path', default='./checkpoints')
    parser.add_argument('--model_path', default='./checkpoints')
    parser.add_argument('--prefix', default='llava-RL_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=20)
    parser.add_argument('--prefix_length_clip', type=int, default=540)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=4, help='transformer mapper layers')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    
    device = torch.device('cuda:2')
    device_2 = torch.device('cuda:2')
    #clip_model, preprocess = clip.load(args.clip_path, device=device, jit=False)
    #dataset = ClipCocoDataset(args.data, args.img_dir, prefix_length, args.gpt_path, preprocess, normalize_prefix=args.normalize_prefix)
    
    data_args = DataArguments(
        data_path = r'/data16tb/fmy/LLaVA-main/playground/data/llava_v1_5_mix665k-img-one.json',
        image_folder = r'/data16tb/fmy/LLaVA-main/playground/data',
        is_multimodal=True,
        image_aspect_ratio = 'pad',
        vision_tower = r'/data16tb/fmy/llava-v1.5-7b/clip-vit-large-patch14-336',
        mm_vision_select_feature = 'cls_patch',
        mm_vision_select_layer = -2,
        mm_use_im_start_end = False,
        mm_use_im_patch_token = False,
        pretrain_mm_mlp_adapter = None
    )
    training_args = TrainingArguments(model_max_length = 2048)
    model_args = ModelArguments(model_name_or_path = r'/data16tb/fmy/llava-v1.5-7b')
    
    vision_tower = CLIPVisionTower(data_args.vision_tower, args=data_args)
    data_args.image_processor = vision_tower.image_processor
    data_args.image_processor_224 = vision_tower.image_processor_224
    data_args.is_multimodal = True
    
    # llava_model = LlavaLlamaForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    # )
    llava_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    llava_tokenizer.pad_token = llava_tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
        
    # llava_model.get_model().initialize_vision_modules(model_args=data_args)
    
    # llava_model.config.image_aspect_ratio = data_args.image_aspect_ratio
    # llava_model.config.tokenizer_padding_side = llava_tokenizer.padding_side
    # llava_model.config.tokenizer_model_max_length = llava_tokenizer.model_max_length
    # llava_model.config.mm_use_im_start_end = data_args.mm_use_im_start_end
    # llava_model.config.mm_use_im_patch_token = data_args.mm_use_im_patch_token
    
    
    
    data_module = make_supervised_data_module(tokenizer=llava_tokenizer, data_args=data_args)
    train_dataset = data_module['train_dataset']
    data_collator = data_module['data_collator']
    
    dataloader_params = {
        "batch_size": args.bs,   
        "collate_fn": data_collator,
        "num_workers": 4,
        "pin_memory": True
    }

    sampler = LengthGroupedSampler(
        dataloader_params['batch_size'],
        world_size= 1,                       #self.args.world_size * self.args.gradient_accumulation_steps,
        lengths=train_dataset.modality_lengths,
        group_by_modality=True,
    )
    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = sampler
        dataloader_params["drop_last"] = False
        dataloader_params["worker_init_fn"] = seed_worker
    
    dataloader = DataLoader(train_dataset, **dataloader_params)
    
    prefix_dim = 640 if args.is_rn else 768
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    
    if args.only_prefix:
        model = ClipCaptionPrefix(args.gpt_path, prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(args.gpt_path, prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_path)
    
    # selector_weight = "/public/home/renwu04/fmy/data/longclip-L.pt"
    # selector, _ = longclip.load(selector_weight, device=device)

    # selector_mlp = TransformerSelector(embedding_dim=768, num_layers=4, num_heads=8, dim_feedforward=1536)
    
    train(dataloader, model, vision_tower, llava_tokenizer, gpt2_tokenizer, args, output_dir=args.out_dir, output_prefix=args.prefix, device=device, device_llava=device_2)


if __name__ == '__main__':
    main()
