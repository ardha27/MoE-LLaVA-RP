import os
import torch
import requests
import base64
from PIL import Image
from io import BytesIO

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger

from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from schemas.input import INPUT_SCHEMA


disable_torch_init()


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def load_image(data):
    # Case 1: If the input is a URL, fetch the image and convert it to a PIL Image
    if data.startswith('http://') or data.startswith('https://'):
        response = requests.get(data)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Case 2: If the input is base64, decode it and convert to a PIL Image
    elif data.startswith('data:image'):
        # Find the start of the base64 string
        base64_str_index = data.find('base64,') + 7
        # Extract the base64 string and decode it
        image_data = base64.b64decode(data[base64_str_index:])
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data)).convert('RGB')
    
    # Case 3: Assume the input is a file path and open the image
    else:
        image = Image.open(data).convert('RGB')
    
    return image


def run_inference(data: dict, current_model_path: str, tokenizer, model, image_processor):
    model_path = data.get('model_path')
    model_name = get_model_name_from_path(model_path)

    if current_model_path != model_path:
        current_model_path = model_path
        tokenizer, model, processor, _ = load_pretrained_model(
            current_model_path,
            None,
            model_name,
            data['load_8bit'],
            data['load_4bit'],
            device='cuda'
        )
        image_processor = processor['image']

    conv_mode = data['conv_mode']  # phi or qwen or stablelm
    conv = conv_templates[conv_mode].copy()
    image = load_image(data['image'])
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + '\n' + data['prompt']
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=data['temperature'],
            max_new_tokens=data['max_new_tokens'],
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return outputs


def handler(job):
    validated_input = validate(job['input'], INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {
            'error': validated_input['errors']
        }

    try:
        payload = validated_input['validated_input']

        outputs = run_inference(
            {
                'model_path': payload.get('model_path'),
                'image': payload.get('image'),
                'prompt': payload.get('prompt'),
                'conv_mode': payload.get('conv_mode'),
                'temperature': payload.get('temperature'),
                'max_new_tokens': payload.get('max_new_tokens'),
                'load_8bit': payload.get('load_8bit'),
                'load_4bit': payload.get('load_4bit')
            },
            CURRENT_MODEL_PATH,
            tokenizer,
            model,
            image_processor
        )

        return {
            'response': outputs
        }
    except Exception as e:
        return {
            'error': str(e)
        }


# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    INITIAL_MODEL_PATH = os.getenv('MODEL', 'LanguageBind/MoE-LLaVA-Phi2-2.7B-4e')
    CURRENT_MODEL_PATH = INITIAL_MODEL_PATH
    logger = RunPodLogger()

    # Model
    model_name = get_model_name_from_path(INITIAL_MODEL_PATH)
    logger.info(f'Loading model: {model_name}')

    tokenizer, model, processor, _ = load_pretrained_model(
        INITIAL_MODEL_PATH,
        None,
        model_name,
        False,
        False,
        device='cuda'
    )
    image_processor = processor['image']

    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
