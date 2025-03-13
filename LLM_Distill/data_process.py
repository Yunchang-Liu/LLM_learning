import os
import json
import http.client
from openai import OpenAI
from tqdm import tqdm

# 使用腾讯云 DeepSeek-R1 接口
os.environ['API_KEY'] = 'sk-das6aya4qpymnue7'

# 设置OpenAI API客户端
client = OpenAI(
    api_key="sk-1skMTBHyBIcKjMNRI2dYdrkwGKcLX0ieA8708lBI49bRFLAh", 
    base_url="https://api.lkeap.cloud.tencent.com/v1",
)

def riddle_sense_map(x):
    # 获取题目和选项
    question = x["question"]["stem"]  # 问题部分
    choices = x["question"]["choices"]  # 选项部分
    full_question = question  # 初始化完整问题为问题stem

    # 将每个选项的标签和内容加入问题中
    labels = [choice["label"] for choice in choices]
    texts = [choice["text"] for choice in choices]
    
    for label, text in zip(labels, texts):
        full_question += f"\n{label}: {text}"  # 选项格式化加入问题中

    return {"question": full_question, "answer": x["answerKey"]}


def reason_with_api(full_question):
    """调用API服务来获取推理和答案"""
    
    # 构建请求的 payload 数据
    payload = json.dumps({
        "model": "deepseek-r1",
        "messages": [
            {"role": "user", "content": f"{full_question}\nPlease think step by step, and put the final answer into \\boxed{{}}."}
        ]
    })

    headers = {
        'Content-Type': "application/json",
        'Authorization': f"Bearer {os.environ.get('API_KEY')}"  # 从环境变量获取API密钥
    }

    # 建立与API服务的连接
    conn = http.client.HTTPSConnection("cloud.infini-ai.com")
    
    # 发送POST请求
    conn.request("POST", "/maas/v1/chat/completions", payload, headers)

    # 获取响应
    res = conn.getresponse()
    data = res.read()
    
    # 解析返回的JSON数据
    response_data = json.loads(data.decode("utf-8"))
    
    # 提取推理内容和最终答案
    reasoning_content = response_data["choices"][0]["message"].get("reasoning_content", "No reasoning available")
    answer = response_data["choices"][0]["message"].get("content", "No answer provided")

    return reasoning_content, answer


def reason_with_deepseek(full_question):
    """调用DeepSeek-R1接口获取思考过程和答案"""
    # 调用接口生成思考过程和答案
    completion = client.chat.completions.create(
        model="deepseek-r1",
        temperature=0.6,
        messages=[
            {'role': 'user', 'content': f"{full_question}\nPlease think step by step, and put the final answer into \\boxed{{}}."}
        ]
    )

    # 提取思考过程和答案
    reasoning_content = completion.choices[0].message.reasoning_content
    answer = completion.choices[0].message.content

    return reasoning_content, answer


def prepare_instruction_data(file_path, output_file, batch_size=1):
    """从jsonl文件读取数据并转换为指令微调格式，每处理batch_size条数据就写入一次"""

    # 存储当前批次的指令数据
    instruction_data_list = []

    # 读取jsonl文件
    with open(file_path, "r") as file:
        for line in tqdm(file, desc="Processing", unit="question"):
            try:
                entry = json.loads(line)  # 解析每行的json数据

                # 取出完整问题
                full_question = entry["question"]["stem"]
                choices = entry["question"]["choices"]
                labels = [choice["label"] for choice in choices]
                texts = [choice["text"] for choice in choices]

                for label, text in zip(labels, texts):
                    full_question += f"\n{label}: {text}"  # 选项格式化加入问题中
                
                # 调用DeepSeek-R1获取思考过程和答案
                reasoning_content, answer = reason_with_api(full_question)

                # 构建指令微调数据集格式
                instruction_data = {
                    "instruction": full_question,  # 完整问题作为instruction
                    "input": "",  # 输入为空
                    "output": f"{reasoning_content} {answer}"  # 输出是答案
                }

                # 添加到当前批次的数据列表中
                instruction_data_list.append(instruction_data)

                # 每处理完batch_size条数据就写入文件
                if len(instruction_data_list) >= batch_size:
                    with open(output_file, "a", encoding="utf-8") as out_file:
                        for item in instruction_data_list:
                            json.dump(item, out_file, ensure_ascii=False)
                            out_file.write("\n")  # 每次写入后换行

                    # 清空当前批次的数据列表
                    instruction_data_list = []

            except Exception as e:
                print(f"Error processing entry: {e}")
                continue  # 跳过当前条目，继续处理下一个

    # 如果剩下的数据不足一个batch_size，也要写入文件
    if instruction_data_list:
        with open(output_file, "a", encoding="utf-8") as out_file:
            for item in instruction_data_list:
                json.dump(item, out_file, ensure_ascii=False)
                out_file.write("\n")





if __name__ == "__main__":
    # 读取本地jsonl文件并转换为指令微调数据
    output_file = "R1_train.jsonl"
    formatted_data = prepare_instruction_data("./riddlesense_dataset/train.jsonl", output_file)