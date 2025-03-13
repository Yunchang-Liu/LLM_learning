table_name_list = ['基金基本信息','基金股票持仓明细','基金债券持仓明细','基金可转债持仓明细','基金日行情表','A股票日行情表','港股票日行情表','A股公司行业划分表','基金规模变动表','基金份额持有人结构']
n = 5
deny_list = ['0','1','2','3','4','5','6','7','8','9','，','？','。',
             '一','二','三','四','五','六','七','八','九','零','十',
            '的','小','请','.','?','有多少','帮我','我想','知道',
             '是多少','保留','是什么','-','(',')','（','）','：',
              '哪个','统计','且','和','来','请问','记得','有','它们']


import csv
import pandas as pd 
import numpy as np
import sqlite3
from tqdm import tqdm
import os
import re
from langchain.utilities import SQLDatabase
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

os.environ["CUDA_VISIBLE_DEVICES"]="0"  # 设置只使用GPU:3

db = SQLDatabase.from_uri("sqlite:///./bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db", sample_rows_in_table_info=2)
db_content = db.table_info

tables = db_content.split('CREATE TABLE')[1:]
for i in range(len(tables)):
    tables[i] = 'CREATE TABLE' + tables[i]

table_info_dict = {}
for table_sample in tables:
    for name in table_name_list:
        if name in table_sample:
            table_info_dict[name] = table_sample


question_csv_file_dir = "./data/01_question_classify.csv"
question_csv_file = pd.read_csv(question_csv_file_dir, delimiter=",", header=0)
model_dir = './Tongyi-Finance-14B-Chat-Int4'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()    
model.generation_config = GenerationConfig.from_pretrained(model_dir,
                                                           trust_remote_code=True,
                                                           temperature = 0.0001,
                                                           top_p = 1,
                                                           do_sample = False,
                                                           seed = 1234)

print('Tongyi-Finance-14B-Chat-Int4 loaded successfully. ✅')


def construct_prompt(question, index_list):
    Examples = '以下是一些例子：'

    for index in index_list:
        Examples += f"问题：{example_question_list[index]}\n"
        Examples += f"SQL：{example_sql_list[index]}\n"

    prompt = "你是一个精通SQL语句的程序员。我会给你一个问题，请按照问题描述，仿照以下例子写出正确的SQL代码。\n"
    
    prompt += Examples
    prompt += f"问题：{question}\n"
    prompt += "SQL："

    return prompt


SQL_examples_file_dir = "./data/ICL_EXP.csv"
SQL_examples_file = pd.read_csv(SQL_examples_file_dir, delimiter = ",", header = 0)

deny_token_list = list()
for word in deny_list:
    temp_tokens = tokenizer(word)['input_ids']
    deny_token_list = deny_token_list + temp_tokens

# 提取问题和SQL列 生成列表
example_question_list = SQL_examples_file['问题'].tolist()
example_sql_list = SQL_examples_file['SQL'].tolist()

# 处理 token，并去除 deny_token_list 里的无效 token
example_token_list = [
    [token for token in tokenizer(q)['input_ids'] if token not in deny_token_list]
    for q in example_question_list
]

# 输出 CSV 文件路径
output_file = './data/03_question_SQL.csv'

# 正则匹配 8 位日期格式
pattern_date = r'\d{8}'

# 打开 CSV 文件进行写入
with open(output_file, 'w', newline='', encoding='utf-8') as g:
    csvwriter = csv.writer(g)
    csvwriter.writerow(['问题id', '问题', 'SQL语句', 'prompt'])

    for _, row in tqdm(question_csv_file.iterrows(), total=len(question_csv_file), desc="Processing Questions"):
        response2, prompt2 = 'N_A', 'N_A'

        if row['分类'] == 'SQL' and row['问题id'] not in [174]:
            temp_question = row['问题']

            # 去掉日期信息（用空格替换匹配到的日期）
            temp_question_for_search = re.sub(pattern_date, ' ', temp_question)

            # Tokenizer 处理 & 过滤停用词
            temp_tokens = tokenizer(temp_question_for_search)['input_ids']
            temp_tokens = [token for token in temp_tokens if token not in deny_token_list]

            # 计算与 SQL 例子的问题的相似度
            similarity_list = [
                len(set(temp_tokens) & set(example_tokens)) / (len(set(temp_tokens)) + len(set(example_tokens)))
                for example_tokens in example_token_list
            ]

            # 选择 n 个最相似的 SQL 示例
            max_indices = np.argsort(similarity_list)[-n:][::-1]  # 取最大 n 个索引，并按相似度降序排列

            # 生成 prompt，只加入长度合适的示例
            total_length = 0
            selected_indices = []
            for index in max_indices:
                new_content_length = len(example_question_list[index]) + len(example_sql_list[index])
                
                if total_length + new_content_length > 2300:
                    break  # 超出长度限制，停止添加
                
                selected_indices.append(index)
                total_length += new_content_length  # 更新累积长度
        
            # 生成最终的 prompt 并调用模型
            prompt2 = construct_prompt(temp_question, selected_indices)
            response2, history = model.chat(tokenizer, prompt2, history=None)

        # 写入 CSV 文件
        csvwriter.writerow([row['问题id'], row['问题'], response2, prompt2])


print("Processing completed ✅")



