import argparse
import torch
import os
import json
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import torchvision.transforms as T
import shortuuid
import math

# 直接从 model_vqa_loader.py 复制需要的函数和变量
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def expand2square_local(pil_img, image_processor):
    background_color = tuple(int(x*255) for x in image_processor.image_mean)
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True, help="Path to the input image")
    parser.add_argument("--question", type=str, required=True, help="Question to ask about the image")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--add_proto", type=str, default='false')
    parser.add_argument("--proto_num", type=int, default=5)
    args = parser.parse_args()

    # 检查图片文件是否存在
    if not os.path.exists(args.image_file):
        print(f"错误: 图片文件不存在: {args.image_file}")
        return

    # 初始化模型
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    # 设置模型配置
    model.post_config(args)
    model.to(dtype=torch.bfloat16)
    
    # 加载 triplet 特征
    triplet_feat_saved = '/home/cjb/data/reltr_ckpt/triplet_feat_6.pt'
    if os.path.exists(triplet_feat_saved):
        model.model.vision_tower.triplet_feat_set = torch.load(triplet_feat_saved)
        print(f"Loaded triplet features: {len(model.model.vision_tower.triplet_feat_set.keys())}")
    else:
        print(f"Warning: Triplet feature file not found at {triplet_feat_saved}")
        model.model.vision_tower.triplet_feat_set = {}

    # 准备输入数据
    image = Image.open(args.image_file).convert('RGB')
    
    # 构建对话
    qs = args.question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 处理图像
    image_tensor = process_images([image], image_processor, model.config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    
    # 准备场景图生成需要的图像
    pil_image = expand2square_local(image, image_processor)
    pil_image = transform(pil_image).unsqueeze(0)

    # 移动到GPU
    input_ids = input_ids.to(device='cuda', non_blocking=True)
    image_tensor = image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
    pil_image = pil_image[0].to(device='cuda', non_blocking=True)

    # 生成回答
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.unsqueeze(0),
            [[prompt]],
            images=image_tensor.unsqueeze(0),
            images_224=None,
            img_path=[args.image_file],
            pil_image=pil_image,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)

    # 解码输出
    input_token_len = input_ids.shape[0]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    # 输出结果
    print("\n" + "="*50)
    print(f"图片: {args.image_file}")
    print(f"问题: {args.question}")
    print(f"回答: {outputs}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()