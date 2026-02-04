#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import ipdb

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class CrossAttention_Proto(nn.Module):

    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads, num_query_token, attention_bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_query_token = num_query_token
        self.initializer_range = 0.02

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_qformer = nn.Sequential(nn.LayerNorm(q_dim), nn.Linear(q_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.k_qformer = nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.v_qformer = nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.o_qformer = nn.Linear(self.num_heads * self.head_dim, q_dim, bias=attention_bias)

        
        nn.init.constant_(self.q_qformer[0].bias, 0)
        nn.init.constant_(self.q_qformer[0].weight, 1.0)
        self.q_qformer[1].weight.data.normal_(mean=0.0, std=self.initializer_range)
        nn.init.constant_(self.k_qformer[0].bias, 0)
        nn.init.constant_(self.k_qformer[0].weight, 1.0)
        self.k_qformer[1].weight.data.normal_(mean=0.0, std=self.initializer_range)
        nn.init.constant_(self.v_qformer[0].bias, 0)
        nn.init.constant_(self.v_qformer[0].weight, 1.0)
        self.v_qformer[1].weight.data.normal_(mean=0.0, std=self.initializer_range)
        self.o_qformer.weight.data.normal_(mean=0.0, std=self.initializer_range)

        if(self.num_query_token):
            self.query_tokens = nn.Parameter(
                torch.zeros(1, self.num_query_token, q_dim)
            )
            self.query_tokens.data.normal_(mean=0.0, std=self.initializer_range)

        ## ipdb.set_trace()
        #self.apply(self._init_weights)
    
    def _init_weights(self, m):
        ## ipdb.set_trace()
        if isinstance(m, nn.Linear):
            #trunc_normal_(m.weight, std=.02)
            torch.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        #elif isinstance(m, nn.Parameter):
        #    normal_(m.data, std=.02)

    def forward(
        self,
        vision_latents, queries, attention_mask=None
    ):
        
        bsz, q_len, _ = queries.size()
        bsz, v_len, _ = vision_latents.size()

        if(self.num_query_token):
            query_tokens = self.query_tokens.expand(bsz, -1, -1)
            queries = queries + query_tokens

        query_states = self.q_qformer(queries)
        key_states = self.k_qformer(vision_latents)
        value_states = self.v_qformer(vision_latents)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)


        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, v_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, v_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim)

        attn_output = self.o_qformer(attn_output)

        return attn_output



class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.tokenizer = AutoTokenizer.from_pretrained('/data/fmy/llava-v1.5-7b', use_fast=False)
        
        self.null_triplet = 0

        # Initialize weights and apply final processing
        self.post_init()
        
    def post_config(self, args):
        self.add_proto = args.add_proto
        self.proto_num = args.proto_num
        print('add_proto: ', self.add_proto)
        print('proto_num: ', self.proto_num)

    def load_qformer_weights(self, model_args):
        ## ipdb.set_trace()
        pretrain_qformer = model_args.pretrain_qformer
        qformer_weights_all = torch.load(pretrain_qformer, map_location='cpu')

        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

        self.qformer_2.load_state_dict(get_w(qformer_weights_all, 'qformer_2'))
        print('Q-Former successfully load weights.')

    def load_gate_weights(self, model_args):
        ## ipdb.set_trace()
        pretrain_gate = model_args.pretrain_gate
        gate_weights = torch.load(pretrain_gate, map_location='cpu')

        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        
        self.gate_fc.load_state_dict(get_w(gate_weights, 'gate_fc'))
        print('Gate_fc successfully load weights.')


    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        #input_ids: list = None,                # in training
        prompt: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        #labels: Optional[list] = None,         # in training
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_224: Optional[torch.FloatTensor] = None,
        img_path: Optional[tuple[str]] = None,
        pil_image: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        step: Optional[int] = None,
        #tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        image_id_pos = torch.where(input_ids==IMAGE_TOKEN_INDEX)   
        # (tensor([0, 1, 2, 3], device='cuda:0'), tensor([35, 35, 35, 35], device='cuda:0'))

        first_time = False
        if input_ids.shape[1] > 1:
             first_time = True

        ## ipdb.set_trace()
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                proto_indice
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_224,
                img_path,
                pil_image,
                #self.qformer_2,
                prompt
            )

        if first_time :
            self.proto_indice = proto_indice
            self.image_id_pos = image_id_pos
        
        ## ipdb.set_trace()

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, input_ids, prompt, past_key_values=None, inputs_embeds=None, **kwargs):
        ## ipdb.set_trace()
        images = kwargs.pop("images", None)
        images_224 = kwargs.pop("images_224", None)
        img_path = kwargs.pop("img_path", None)
        pil_image = kwargs.pop("pil_image", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if images_224 is not None:
            _inputs['images_224'] = images_224
        if img_path is not None:
            _inputs['img_path'] = img_path
        if pil_image is not None:
            _inputs['pil_image'] = pil_image

        _inputs['prompt'] = prompt

        return _inputs    # dict_keys(['input_ids', 'position_ids', 'past_key_values', 'use_cache', 'attention_mask', 'images'])

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
