# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# 检测可用设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 根据设备选择数据类型
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = AutoModel.from_pretrained('/Users/ruanxiaoyang/Desktop/repo/finetune-minicpm/models/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch_dtype) # sdpa or flash_attention_2, no eager

# 将模型移动到设备上（如果是CPU则不需要.cuda()）
model = model.eval().to(device)
tokenizer = AutoTokenizer.from_pretrained('/Users/ruanxiaoyang/Desktop/repo/finetune-minicpm/models/MiniCPM-V-2_6', trust_remote_code=True)

image = Image.open('/Users/ruanxiaoyang/Desktop/repo/finetune-minicpm/dataset/helmet_sample_output_crops/head_helmet_crops/000009_head_0.jpg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(res)

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
