import torch
from tqdm import tqdm  # 导入tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft import LoraConfig, TaskType
import json
import re


# 手动计算准确率
def calculate_accuracy(true_labels, predictions):
    correct = 0
    total = len(true_labels)
    
    for true, pred in zip(true_labels, predictions):
        if true == pred:
            correct += 1
    
    accuracy = correct / total
    return accuracy


# 加载模型和tokenizer
model_path = '/data/Messi/replication/transformer_learning/LLM_learning/Qwen2.5-3B-Instruct'
lora_path = '/data/Messi/projectWork/LLM_project/LLM_Distill/saves/Qwen2.5-3B-Instruct-Lora'

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1 # Dropout 比例
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

# 从测试集文件读取数据
test_data_path = './riddlesense_dataset/dataset/test.json'  # 假设测试集路径为 'test_data.json'

with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 用于保存推理结果
responses = []
true_answers = []

# 遍历测试集并推理
for sample in tqdm(test_data, desc="Processing samples", unit="sample"):
    instruction_text = sample['instruction']
    input_text = sample['input']
    
    # 拼接 prompt
    prompt = instruction_text + input_text
    messages = [{"role": "user", "content": f"{prompt}\nPlease think step by step, and put the final answer into \\boxed{{}}."}]
    
    # 构造文本并转换为模型输入格式
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    # 生成模型输出
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码生成的答案
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 提取模型预测的答案（假设答案字母出现在最后）
    match = re.search(r'\\boxed{(.*?)}', response)
    if match:
        predicted_answer = match.group(1)  # 提取括号内的内容
    else:
        predicted_answer = None  # 如果没有找到匹配的答案

    responses.append(predicted_answer)
    
    # 提取测试集中的真实答案，格式为 \boxed{D}，提取 D
    true_answer = sample['output']
    
    true_answers.append(true_answer)
    

# 评估准确率
accuracy = calculate_accuracy(true_answers, responses)
print(f"Accuracy: {accuracy * 100:.2f}%")