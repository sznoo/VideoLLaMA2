import os

os.environ["FLASH_ATTENTION_2_ENABLED"] = "0"
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"


import sys

sys.path.append("./")
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import time
import torch


def set_model():
    disable_torch_init()
    model_path = "DAMO-NLP-SG/VideoLLaMA2.1-7B-16F"
    model, processor, tokenizer = model_init(model_path)
    return model, processor, tokenizer


def inference(modal, modal_path, instruct, model, processor, tokenizer):

    # Base model inference (only need to replace model_path)
    # model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base'
    if modal == "text":
        output = mm_infer(
            torch.zeros(1),
            instruct,
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            modal=modal,
        )

    else:
        output = mm_infer(
            processor[modal](modal_path),
            instruct,
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            modal=modal,
        )
    print("=" * 100)
    print(output)


if __name__ == "__main__":
    modal = "text"
    modal_path = "assets/cat_and_chicken.mp4"
    # instruct = 'Generate 10 unique captions for the provided video frame, each from a different perspective.'
    instruct = "Request: Please rewrite the following instruction to make it clearer and more detailed: [Generate 10 unique captions for the provided video frame, each from a different perspective.]"
    model, processor, tokenizer = set_model()
    # device = torch.device("cpu")
    # model.to(device)
    t = time.time()
    inference(modal, modal_path, instruct, model, processor, tokenizer)
    print(time.time() - t)
