import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

sys.path.append("Reefknot/LLaVA")
import torch.nn.functional as F
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.DTC import DTC_function
from PIL import Image
import math

DTC_function()

def get_path(image_id):
    Image_path1 = '/hpc2hdd/home/yuxuanzhao/lijungang/nk_code/LLM/Datasets/VG_dataset/VG_100K'
    Image_path2 = '/hpc2hdd/home/yuxuanzhao/lijungang/nk_code/LLM/Datasets/VG_dataset/VG_100K_2'
    # if image is not None:
    image_id = str(image_id)
    if image_id.endswith('.jpg'):
        image_id = image_id.split('.')[0]
    if os.path.exists(os.path.join(Image_path1, image_id+'.jpg')):
        # print('Find image in VG100K(small one!) image path is:',os.path.join(Image_path1, image_id+'.jpg'))
        return os.path.join(Image_path1, image_id+'.jpg')
    elif os.path.exists(os.path.join(Image_path2, image_id+'.jpg')):
        return os.path.join(Image_path2, image_id+'.jpg')
    else:
        print('Cannot find image {}.jpg'.format(image_id))
        return None


def entropy_vector(prob_vector):
    """
    Calculate the entropy of a probability vector.

    Args:
    prob_vector (list or torch.Tensor): A vector representing a probability distribution.

    Returns:
    float: The entropy of the probability vector.
    """
    prob_tensor = torch.tensor(prob_vector, dtype=torch.float32)

    # To avoid log(0), we mask zero values
    mask = prob_tensor > 0
    prob_tensor = prob_tensor[mask]
    
    # Calculate entropy
    entropy_value = -torch.sum(prob_tensor * torch.log2(prob_tensor))

    return entropy_value.item()





def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        image_file = line["image_id"] + ".jpg"
        # image_path = os.path.join(args.image_folder, image_file)
        image_path = get_path(line["image_id"])
        if not os.path.exists(image_path):
            print(f"Image file {image_file} not found, skipping.")
            continue 

        qs = line["query_prompt"]
        label = line["label"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            layer_score, output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=False, # True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=2,
                use_cache=True,
                output_scores=True,
                apha=args.apha,
                threshold=args.threshold,
                layer=args.layer,
            )
        mllm = args.model_path.split('/')[-1]
        outputs = tokenizer.batch_decode(
            output_ids.sequences, skip_special_tokens=True
        )[0].strip()
        ans_file.write(
            json.dumps(
                {
                    "image_id": line["image_id"],
                    "query_prompt": cur_prompt,
                    "response": outputs,
                    "label": label,
                    "mllm_name": mllm
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--apha", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--layer", type=int, default=38)
    args = parser.parse_args()

    eval_model(args)
