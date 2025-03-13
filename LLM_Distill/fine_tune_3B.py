from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset
import os

# 配置 LoRA 微调
def setup_lora(model_name, lora_r=8, lora_alpha=32, target_modules=None, lora_dropout=0.1):
    # 设定 LoRA 配置
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules if target_modules else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 选择 LoRA 微调的层
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM
    )
    
    # 加载模型并应用 LoRA
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())  # 打印可训练参数
    return model

def train_model():
    # 加载模型和tokenizer
    model_name = "/root/root/autodl-tmp/qwen/Qwen2.5-3B-Instruct"  # 使用 Qwen2.5-3B 模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 使用 LoRA 微调模型
    model = setup_lora(model_name)

    # 设置训练参数
    args = TrainingArguments(
        output_dir='./results/Qwen2.5-3B-Instruct-Lora', 
        num_train_epochs=1, 
        do_train=True, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        logging_steps=100,
        save_strategy='epoch',
        save_total_limit=10,
        bf16=True,
        learning_rate=0.0005,
        lr_scheduler_type='cosine',
        dataloader_num_workers=8,
        dataloader_pin_memory=True
    )

    # 加载训练数据集
    dataset = SFTDataset('./riddlesense_dataset/dataset/train.json', tokenizer=tokenizer, max_seq_len=2500)  # 假设数据集文件是 `train_data.json`

    # 设置数据集和训练器
    data_collator = DefaultDataCollator()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model('./saves/Qwen2.5-3B-Instruct-Lora')
    trainer.save_state()

if __name__ == "__main__":
    train_model()
