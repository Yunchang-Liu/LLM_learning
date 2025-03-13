import json
import random
import re


def process_train_data(input_file, output_file, max_length=1000, max_output_length=5500):
    """
        从经过DeepSeek-R1得到的推理数据集中,筛选得到最后的1000条训练集 (长度不能过长,答案要符合格式,TODO:这里没有验证答案是否正确)
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 用于保存格式化后的数据
    formatted_data = []

    # 处理每个数据样本
    for sample in data:
        output_text = sample['output']

        # 确保 output 字段符合 \boxed{X} 的格式
        if re.search(r'\\boxed{(.*?)}', output_text):
            # 检查 output 字符串长度
            if len(output_text) < max_output_length:
                formatted_data.append(sample)

        # 如果已经达到最大条数，提前停止
        if len(formatted_data) >= max_length:
            break

    # 将格式化后的数据保存为 json 格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)


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

    return {"instruction": full_question, "input": "", "output": x["answerKey"]}



def process_test_data(input_file, output_file, sample_size=500): 
    """
        从测试集中随机采样500条,作为最终的test.json
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    sampled_data = random.sample(data, sample_size)

    formatted_data = []
    # 处理每个数据样本
    for sample in sampled_data:
        # 使用 riddle_sense_map 函数进行处理
        formatted_sample = riddle_sense_map(sample)
        formatted_data.append(formatted_sample)

    # 将格式化后的数据保存为 json 格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # 输入输出文件路径
    input_train_file = './R1_train_merged.jsonl'  # 原始数据路径
    output_train_file = './dataset/train.json'  # 保存转换后数据的路径
    process_train_data(input_train_file, output_train_file)


    # 输入输出文件路径
    input_test_file = 'test.jsonl'  # 原始数据路径
    output_test_file = './dataset/test.json'  # 保存转换后数据的路径
    process_test_data(input_test_file, output_test_file)
