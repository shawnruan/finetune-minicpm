# test_cpu.py - CPU only inference
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

print("Loading model for CPU inference...")

model = AutoModel.from_pretrained(
    '/Users/ruanxiaoyang/Desktop/repo/finetune-minicpm/models/merge-lora', 
    trust_remote_code=True,
    torch_dtype=torch.float32, 
    device_map="cpu"
)

model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    '/Users/ruanxiaoyang/Desktop/repo/finetune-minicpm/models/merge-lora', 
    trust_remote_code=True
)

print("Model loaded successfully on CPU")

# 检查图片文件是否存在
image_path = '/Users/ruanxiaoyang/Desktop/repo/finetune-minicpm/dataset/helmet_sample_output_crops/head_helmet_crops/000009_head_0.jpg'
try:
    image = Image.open(image_path).convert('RGB')
    print(f"Image loaded: {image.size}")
except FileNotFoundError:
    print(f"Image not found at {image_path}")
    print("Please check the image path or use a different image")
    exit()

question = 'What is in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]

print("Starting inference...")
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print("Response:", res)

print("\nTesting streaming response...")
## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    stream=True
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')

print(f"\nComplete response: {generated_text}") 