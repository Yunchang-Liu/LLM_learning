import os
import csv
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


os.environ["CUDA_VISIBLE_DEVICES"]="0"  # 设置只使用GPU:3

# 设定参数
n = 4
pattern1 = r'\d{8}'  # 正则匹配8位数字（日期格式）

# 定义停用词列表
deny_list = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '？', '。',
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '零', '十',
    '的', '小', '请', '.', '?', '有多少', '帮我', '我想', '知道',
    '是多少', '保留', '是什么', '-', '(', ')', '（', '）', '：',
    '哪个', '统计', '且', '和', '来', '请问', '记得', '有', '它们'
]

# 读取SQL执行结果 和 问题分类
data_file = pd.read_csv('./data/04_question_SQL_exed.csv', delimiter=",")
data_file2 = pd.read_csv('./data/01_question_classify.csv', delimiter=",")

# 加载大模型
model_dir = './Tongyi-Finance-14B-Chat-Int4'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, temperature=0.00001, top_p=1, do_sample=False, seed=1234)


# 预处理停用词 token
deny_token_list = []
for word in deny_list:
    deny_token_list.extend(tokenizer(word)['input_ids'])

# 读取SQL示例文件
SQL_examples_file = pd.read_csv("./data/ICL_EXP.csv", delimiter=",")
example_question_list = SQL_examples_file['问题'].tolist()  # 示例问题
example_data_list = SQL_examples_file['资料'].tolist()  # 示例执行结果
example_FA_list = SQL_examples_file['FA'].tolist()  # 示例最终回答

# 处理 token，并去除停用词
example_token_list = [
    [token for token in tokenizer(q)['input_ids'] if token not in deny_token_list]
    for q in example_question_list
]

# 定义 Prompt 生成函数
def generate_prompt(question, data, index_list):
    """ 生成 prompt 以便模型生成答案 """
    examples = '\n'.join([
        f"问题：{example_question_list[i]}\n资料：{example_data_list[i]}\n答案：{example_FA_list[i]}"
        for i in index_list
    ])
    return f"""
        你要进行句子生成工作，根据提供的资料来回答对应的问题。
        下面是一些例子。注意问题中对小数位数的要求。

        {examples}
        
        问题：{question}
        资料：{data}
        答案：
    """

# 结果文件路径
output_file = './data/05_FA_SQL.csv'

# 打开 CSV 文件进行写入
with open(output_file, 'w', newline='', encoding='utf-8') as g:
    csvwriter = csv.writer(g)
    csvwriter.writerow(['问题id', '问题', 'FA', 'SQL结果'])

    for i in tqdm(range(len(data_file)), desc="Processing Queries"):
        temp_question = data_file.loc[i, '问题']
        class_ans = data_file2.loc[i, '分类']
        SQL_search_result = data_file.loc[i, '执行结果']
        temp_FA = 'N_A'  # 默认回答

        # 如果不是 SQL 问题，则跳过
        if class_ans != 'SQL':
            SQL_search_result = 'N_A'
        elif SQL_search_result != 'N_A' and len(SQL_search_result) > 0:
            # 限制 SQL 结果长度
            SQL_search_result = SQL_search_result[:250]

            # 去掉日期信息（用空格替换匹配到的日期）
            temp_question_clean = re.sub(pattern1, ' ', temp_question)

            # Tokenizer 处理 & 过滤停用词
            temp_tokens = [
                token for token in tokenizer(temp_question_clean)['input_ids']
                if token not in deny_token_list
            ]

            # 计算与 SQL 例子的问题的相似度
            similarity_list = np.array([
                len(set(temp_tokens) & set(example_tokens)) / (len(set(temp_tokens)) + len(set(example_tokens)))
                for example_tokens in example_token_list
            ])

            # 选择 n 个最相似的 SQL 示例
            max_indices = similarity_list.argsort()[-n:][::-1]  # 取最大 n 个索引，并按相似度降序排列

            # 生成 Prompt 并调用模型
            prompt2 = generate_prompt(temp_question, SQL_search_result, max_indices)
            temp_FA, history = model.chat(tokenizer, prompt2, history=None)

        else:
            SQL_search_result = 'SQL未能成功执行！'

        # 写入 CSV 文件
        csvwriter.writerow([data_file.loc[i, '问题id'], temp_question, temp_FA, SQL_search_result])

print("Processing Complete. ✅")
