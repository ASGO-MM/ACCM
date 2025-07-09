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
import json
import clip
import skimage.io as io
from PIL import Image
from clipcap_model_question import ClipCaptionModel, ClipCaptionPrefix, ClipCaptionGPT
from clip_encoder import CLIPVisionTower
from llava_llama import LlavaLlamaForCausalLM
from llava_trainer import LengthGroupedSampler
from llava_datasets import *
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from decoding_scheme import generate_beam
import matplotlib.pyplot as plt
from selector_transformer import TransformerSelector
from model import longclip

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


def train(train_dataloader, model: ClipCaptionModel, vision_tower, llava_model, selector, selector_mlp, gpt2_tokenizer, args,
          lr: float = 1e-6, lr_selector: float = 2e-5, warmup_steps: int = 1000, output_dir: str = ".", output_prefix: str = "", device=torch.device('cuda:0'), device_llava=torch.device('cuda:1')):

    #batch_size = args.bs
    print('lr: ', lr)
    print('lr_selector: ', lr_selector)
    
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = model.to(device)
    model.gpt.train()
    
    for k, v in model.named_parameters():
        if('clip_project' in k):
            v.requires_grad = False
    for k, v in model.named_parameters():
        print(k, v.requires_grad)
    
    vision_tower = vision_tower.to(device, dtype=torch.bfloat16)
    vision_tower.eval()
    vision_tower.requires_grad_(False)
    
    llava_model = llava_model.to(dtype=torch.bfloat16)
    llava_model = llava_model.to(device_llava)
    llava_model.eval()
    llava_model.requires_grad_(False)
    #ipdb.set_trace()
    
    selector = selector.to(device)
    selector.eval()
    selector.requires_grad_(False)

    selector_mlp = selector_mlp.to(device)
    selector_mlp.train()
    
    print()
    for k, v in selector.named_parameters():
        print(k, v.requires_grad)
    for k, v in selector_mlp.named_parameters():
        print(k, v.requires_grad)
    
    
    #optimizer = AdamW(model.gpt.parameters(), lr=lr)
    optimizer = AdamW([
	    {'params': model.gpt.parameters(), 'lr': lr,}, 
	    {'params': selector_mlp.parameters(), 'lr': lr_selector},
	])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    loss_list = []
    selector_loss_list = []
    loss_curve_path = '/home/fmy/LLaVA-PruMerge-text-3/loss_curve/captionModel+selector-mlp_only-gpt_1e-6_2e-5_200k_one_1epoch_question_wo-normalize_loss-weight'
    #criterion_nll = nn.NLLLoss()
    if not os.path.exists(loss_curve_path):
        os.makedirs(loss_curve_path)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, inputs in enumerate(train_dataloader):
            #ipdb.set_trace()
            model.gpt.zero_grad()
            selector_mlp.zero_grad()
            B = inputs['images'].shape[0]
            # process prompt
            prompt = []
            for questions in inputs['prompt']:
                new_questions = []
                assert len(questions) == 1     # for one q-a pairs !!!
                for q in questions:
                    new_questions.append(q.split('\n')[0])
                prompt.append(new_questions)
            #ipdb.set_trace()        
            _, image_features, x_others, supplement_token = vision_tower(inputs['images'].to(dtype=torch.bfloat16), prompt, None)   # inputs['images_224'].to(dtype=torch.bfloat16)
            #ipdb.set_trace()
            generated_text_beam_1 = []
            generated_text_beam_2 = []
            generated_text_beam_3 = []
            for i in range(B):
                prefix = supplement_token[i].unsqueeze(dim=0).to(dtype=torch.float32)
                model.eval()
                with torch.no_grad():
                    prefix_embed = model.clip_project(prefix).reshape(1, model.prefix_length, -1)
                # add question embedding
                #ipdb.set_trace()
                tokens = torch.tensor(gpt2_tokenizer.encode(prompt[i][0]))   # (N)
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)   # torch.Size([1, N, 768])
                # generated = nnf.normalize(generated, dim=2)   # normalize question embedding
                prefix_embed = torch.cat((generated, prefix_embed), dim=1)
                
                caption_list = generate_beam(model, gpt2_tokenizer, embed=prefix_embed, beam_size=3)
                generated_text_beam_1.append(caption_list[0])
                generated_text_beam_2.append(caption_list[1])
                generated_text_beam_3.append(caption_list[2])
            # ref probs
            inputs['x_others'] = image_features.to(device_llava)
            inputs['generated_text'] = []
            inputs['decoding_scheme'] = ''
            with torch.no_grad():
                ref_probs = llava_model(**inputs)['ans_probs']    # []
            # beam_1 cap
            inputs['x_others'] = x_others.to(device_llava)  # (B, ratio, C)
            inputs['generated_text'] = generated_text_beam_1  # []  len=B
            inputs['decoding_scheme'] = 'beam_1'
            with torch.no_grad():
                beam_1_probs = llava_model(**inputs)['ans_probs'] 
            # beam_2 cap
            inputs['x_others'] = x_others.to(device_llava)  # (B, ratio, C)
            inputs['generated_text'] = generated_text_beam_2  # []  len=B
            inputs['decoding_scheme'] = 'beam_2'
            with torch.no_grad():
                beam_2_probs = llava_model(**inputs)['ans_probs'] 
            # beam_3 cap
            inputs['x_others'] = x_others.to(device_llava)  # (B, ratio, C)
            inputs['generated_text'] = generated_text_beam_3  # []  len=B
            inputs['decoding_scheme'] = 'beam_3'
            with torch.no_grad():
                beam_3_probs = llava_model(**inputs)['ans_probs'] 
                
            # process exception
            error_flag = False
            for i in range(B):
                if(beam_1_probs[i].shape != ref_probs[i].shape):
                    error_flag = True
                if(beam_2_probs[i].shape != ref_probs[i].shape):
                    error_flag = True
                if(beam_3_probs[i].shape != ref_probs[i].shape):
                    error_flag = True
            if(error_flag == True):
                print("Shape error!")
                print(inputs['id'])
                for i in range(B):
                    print(ref_probs[i].shape, beam_1_probs[i].shape, beam_2_probs[i].shape, beam_3_probs[i].shape)
                print()
                continue
            
            #ipdb.set_trace()
            flag_kl = []
            weight_kl = []
            
            for i in range(B):
                P_beam_1 = beam_1_probs[i].softmax(dim=-1)
                P_beam_2 = beam_2_probs[i].softmax(dim=-1)
                P_beam_3 = beam_3_probs[i].softmax(dim=-1)
                Q = ref_probs[i] + 1e-10
                Q = Q.softmax(dim=-1)
                
                beam_1_score = nnf.kl_div(P_beam_1.log(), Q, reduction='batchmean')
                beam_2_score = nnf.kl_div(P_beam_2.log(), Q, reduction='batchmean')
                beam_3_score = nnf.kl_div(P_beam_3.log(), Q, reduction='batchmean')
                #ipdb.set_trace()
                
                avg_score = (beam_1_score + beam_2_score + beam_3_score) / 3.0
                margin = torch.tensor([avg_score - beam_1_score, avg_score - beam_2_score, avg_score - beam_3_score])  # 0-beam_1, 1-beam_2, 2-beam_3
                flag = torch.argmax(margin)
                flag_kl.append(flag)
                weight_kl.append(margin[flag])
            
            
            # train captionModel
            tokens = []    
            for i in range(B):
                if(flag_kl[i] == 0):
                    tokens.append(generated_text_beam_1[i])
                if(flag_kl[i] == 1):
                    tokens.append(generated_text_beam_2[i])
                if(flag_kl[i] == 2):
                    tokens.append(generated_text_beam_3[i])
                    
            assert len(tokens) == B
            #ipdb.set_trace()
            tokens_id = []
            for i in range(B):
                tokens_id.append(torch.tensor(gpt2_tokenizer.encode(tokens[i]), dtype=torch.int64))
            #ipdb.set_trace()
            
            tokens = torch.nn.utils.rnn.pad_sequence(
                tokens_id,
                batch_first=True,
                padding_value=-1
            )
            mask = tokens.ge(0)  # mask is zero where we out of sequence
            tokens[~mask] = 0
            mask = mask.float()
            #ipdb.set_trace()
            
            # add loss weight
            loss_weight = torch.sum(torch.tensor(weight_kl))  #
            #ipdb.set_trace()
            loss_weight = (loss_weight / float(B)) * 4   # loss weight
            
            prompt_id = []
            for i in range(B):
                prompt_id.append(torch.tensor(gpt2_tokenizer.encode(prompt[i][0]), dtype=torch.int64))
            prompt_id_pad = torch.nn.utils.rnn.pad_sequence(prompt_id, batch_first=True, padding_value=-1)
            mask_prompt = prompt_id_pad.ge(0)
            prompt_id_pad[~mask_prompt] = 0
            mask_prompt = mask_prompt.float()
            #ipdb.set_trace()
            
            mask = torch.cat((mask_prompt, torch.ones(B, model.prefix_length), mask), dim=1)  # adding prefix mask
            
            model.gpt.train()
            prefix = supplement_token.to(dtype=torch.float32)
            tokens, mask = tokens.to(device), mask.to(device)
            prompt_id_pad = prompt_id_pad.to(device)
            outputs = model(tokens, prefix, prompt_id_pad, mask)
            
            logits = outputs.logits[:, prompt_id_pad.shape[1] + model.prefix_length - 1: -1]
            #ipdb.set_trace()
            loss_caption = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0) * loss_weight
            
            loss_list.append(loss_caption.item())
            
            # train selector
            out_list = []
            label_list = []
            for i in range(B):
                prefix = 'Question: '
                for question in prompt[i]:
                    prefix = prefix + question
                prefix = prefix + '  Supplemental information for questions: '
                prefix = prefix.replace('\nAnswer with the option\'s letter from the given choices directly.', '')
                prefix = prefix.replace('\nAnswer the question using a single word or phrase.', '')
                tokens = []
                tokens.append(prefix + generated_text_beam_1[i])  
                tokens.append(prefix + generated_text_beam_2[i])
                tokens.append(prefix + generated_text_beam_3[i])
                
                text = longclip.tokenize(tokens, truncate=True).to(device)
                with torch.no_grad():
                    text_features = selector.encode_text(text)     # torch.Size([3, 768]), dtype=float16

                #ipdb.set_trace()
                out = selector_mlp(text_features.unsqueeze(dim=0)).squeeze()
                if(flag_kl[i] == 0):
                    new_label = torch.tensor([1, 0, 0]).to(device=device, dtype=out.dtype)
                if(flag_kl[i] == 1):
                    new_label = torch.tensor([0, 1, 0]).to(device=device, dtype=out.dtype)
                if(flag_kl[i] == 2):
                    new_label = torch.tensor([0, 0, 1]).to(device=device, dtype=out.dtype)
                
                #target = torch.argmax(weight_kl[i]).unsqueeze(dim=0).to(device=device, dtype=torch.int64)
                out_list.append(out)
                label_list.append(new_label)
                
                #loss += loss_i
                #ipdb.set_trace()

            #ipdb.set_trace()
            #if((B - skip) != 0):  
            out_tensor = torch.stack(out_list).to(device=device)
            label_tensor = torch.stack(label_list).to(device=device)   
            loss_selector = nnf.kl_div(out_tensor.log(), label_tensor, reduction='batchmean') * loss_weight
            #
            
            selector_loss_list.append(loss_selector.item())
            
            loss = loss_caption + loss_selector
            
            
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss_caption": loss_caption.item(), "loss_selector": loss_selector.item(), "loss_weight": loss_weight})   # "loss_weight": loss_weight
            #progress.set_postfix({"loss_weight": loss_weight})
            progress.update()
            if (idx + 1) % 2000 == 0:
                torch.save(model.state_dict(), os.path.join(output_dir, f"captionModel_{output_prefix}_latest_{epoch+1}_{idx + 1}_iter.pt"),)
                torch.save(selector_mlp.state_dict(), os.path.join(output_dir, f"selector-mlp_{output_prefix}_latest_{epoch+1}_{idx + 1}_iter.pt"),)
            
            if (idx + 1) % 200 == 0:
                plt.figure(figsize=(17, 6))
                plt.subplot(121)
                plt.plot(torch.arange(idx+1), loss_list)
                plt.xlabel("iterations")
                plt.ylabel("loss")
                plt.title("captionModel")
                
                plt.subplot(122)
                plt.plot(torch.arange(idx+1), selector_loss_list)
                plt.xlabel("iterations")
                plt.ylabel("loss")
                plt.title("selector mlp")
                plt.savefig(os.path.join(loss_curve_path, f"{output_prefix}_latest_{epoch+1}_{idx + 1}_iter.jpg"))
                
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), os.path.join(output_dir, f"captionModel_{output_prefix}-{epoch:03d}.pt"),)
            torch.save(selector_mlp.state_dict(), os.path.join(output_dir, f"selector-mlp_{output_prefix}-{epoch:03d}.pt"),)
            
            plt.figure(figsize=(17, 6))
            plt.subplot(121)
            plt.plot(torch.arange(idx+1), loss_list)
            plt.xlabel("iterations")
            plt.ylabel("loss")
            plt.title("captionModel")
            
            plt.subplot(122)
            plt.plot(torch.arange(idx+1), selector_loss_list)
            plt.xlabel("iterations")
            plt.ylabel("loss")
            plt.title("selector mlp")
            plt.savefig(os.path.join(loss_curve_path, f"{output_prefix}_latest_{epoch + 1}_epoch.jpg"))

    return model


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
    parser.add_argument('--only_gpt', dest='only_gpt', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=4, help='transformer mapper layers')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    
    device = torch.device('cuda:1')
    device_2 = torch.device('cuda:0')
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    data_args = DataArguments(
        data_path = r'/home/fmy/data/LLaVA-main/playground/data/llava_v1_5_mix665k-img-one_revision.json',
        image_folder = r'/home/fmy/data/LLaVA-main/playground/data',
        is_multimodal=True,
        image_aspect_ratio = 'pad',
        vision_tower = r'/home/fmy/data/llava-v1.5-7b/clip-vit-large-patch14-336',
        mm_vision_select_feature = 'cls_patch',
        mm_vision_select_layer = -2,
        mm_use_im_start_end = False,
        mm_use_im_patch_token = False,
        pretrain_mm_mlp_adapter = None
    )
    training_args = TrainingArguments(model_max_length = 2048)
    model_args = ModelArguments(model_name_or_path = r'/home/fmy/data/llava-v1.5-7b')
    
    vision_tower = CLIPVisionTower(data_args.vision_tower, args=data_args)
    data_args.image_processor = vision_tower.image_processor
    data_args.image_processor_224 = vision_tower.image_processor_224
    data_args.is_multimodal = True
    
    llava_model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
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
        
    llava_model.get_model().initialize_vision_modules(model_args=data_args)
    
    llava_model.config.image_aspect_ratio = data_args.image_aspect_ratio
    llava_model.config.tokenizer_padding_side = llava_tokenizer.padding_side
    llava_model.config.tokenizer_model_max_length = llava_tokenizer.model_max_length
    llava_model.config.mm_use_im_start_end = data_args.mm_use_im_start_end
    llava_model.config.mm_use_im_patch_token = data_args.mm_use_im_patch_token
    
    
    
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
    
    if args.only_gpt:
        model = ClipCaptionGPT(args.gpt_path, prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only GPT")
    else:
        model = ClipCaptionModel(args.gpt_path, prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_path)
    
    selector_weight = "/home/fmy/data/longclip-L.pt"
    selector, _ = longclip.load(selector_weight, device=device)

    selector_mlp = TransformerSelector(embedding_dim=768, num_layers=4, num_heads=8, dim_feedforward=1536)
    
    train(dataloader, model, vision_tower, llava_model, selector, selector_mlp, gpt2_tokenizer, args, output_dir=args.out_dir, output_prefix=args.prefix, device=device, device_llava=device_2)


if __name__ == '__main__':
    main()
