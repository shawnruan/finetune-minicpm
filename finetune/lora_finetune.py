import glob
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
from torchvision import transforms

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from transformers import AutoModel, AutoTokenizer
from transformers.integrations import deepspeed
from transformers import AutoModel, AutoTokenizer

from dataset import SupervisedDataset, data_collator
from trainer import CPMTrainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="E:\\recongize_helmet\\models\\MiniCPM-V-2_6")


@dataclass
class DataArguments:
    data_path: str = field(
        default="E:\\recongize_helmet\\LLaMA-Factory\\data\\mllm_demo.json", 
        metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default="E:\\recongize_helmet\\LLaMA-Factory\\data\\mllm_demo.json", 
        metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=False)
    llm_type: str = field(default="qwen")
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)
    
    # 训练相关参数
    remove_unused_columns: bool = field(default=False)
    label_names: Optional[List[str]] = field(default_factory=lambda: ["labels"])
    prediction_loss_only: bool = field(default=False)
    bf16: bool = field(default=True)
    bf16_full_eval: bool = field(default=True)
    fp16: bool = field(default=False)
    fp16_full_eval: bool = field(default=False)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    max_steps: int = field(default=10000)
    eval_steps: int = field(default=1000)
    output_dir: str = field(default="output/output_minicpmv26")
    logging_dir: str = field(default="output/output_minicpmv26")
    logging_strategy: str = field(default="steps")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=10)
    learning_rate: float = field(default=1e-6)
    weight_decay: float = field(default=0.1)
    adam_beta2: float = field(default=0.95)
    warmup_ratio: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    deepspeed: Optional[str] = field(default="ds_config_zero3.json")
    report_to: str = field(default="tensorboard")


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir,)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    transform,
    data_collator=None,
    llm_type="qwen",
    slice_config=None,
    patch_size=14,
    query_nums=64,
    batch_vision=False,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json,
            transform,
            tokenizer,
            slice_config=slice_config,
            llm_type=llm_type,
            patch_size=patch_size,
            query_nums=query_nums,
            batch_vision=batch_vision,
            max_length=max_length,
        )
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator= partial(data_collator, max_length=max_length),
    )


def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        
    return {'Total': all_param, 'Trainable': trainable_params}


local_rank = 0


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) : 
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )
    
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
        init_vision=True,
        init_audio=False,
        init_tts=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
    if not training_args.tune_llm:
        model.llm.requires_grad_(False)
        
    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
            
        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-V model.")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        modules_to_save = ['embed_tokens','resampler']
        if training_args.tune_vision:
            modules_to_save.append('vpm')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))

    llm_type = training_args.llm_type    
    
    rank0_print(f'llm_type={llm_type}')

    
    # Load data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        transform=transform_func,
        data_collator=data_collator,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=model.config.patch_size,
        query_nums=model.config.query_num,
        batch_vision=batch_vision,
        max_length=training_args.model_max_length,
    )
    
    training_args.gradient_checkpointing_kwargs={"use_reentrant":False}
    trainer = CPMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()