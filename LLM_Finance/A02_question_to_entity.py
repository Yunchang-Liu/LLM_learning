import csv
import pandas as pd
import numpy as np
from modelscope import AutoTokenizer

# 使用金融大模型的tokenizer进行分词
model_dir = './Tongyi-Finance-14B-Chat-Int4'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 读取问题分类文件
new_question_file = pd.read_csv('./data/01_question_classify.csv', delimiter=",", header=0)

# 读取公司信息文件
company_file = pd.read_csv('./data/pdf_to_company.csv', delimiter=",", header=0)
company_names = company_file['公司名称'].tolist()
company_csv_files = company_file['csv文件名'].tolist()

# 对公司名称进行Tokenizer处理
company_name_tokens = [set(tokenizer(c_name)['input_ids']) for c_name in company_names]

# 计算相似度函数
def compute_similarity(question_tokens, company_tokens):
    """计算 Jaccard 相似度"""
    return len(question_tokens & company_tokens) / (len(question_tokens) + len(company_tokens))

# 处理并写入新 CSV 文件
output_file = './data/02_question_to_entity.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as g:
    csvwriter = csv.writer(g)
    csvwriter.writerow(['问题id', '问题', '分类', '对应实体', 'csv文件名'])

    for _, row in new_question_file.iterrows():
        q_id, question, category = row['问题id'], row['问题'], row['分类']
        matched_entity = 'N_A'
        matched_csv = 'N_A'

        if category == 'Text':
            # 计算问题Tokenizer
            question_tokens = set(tokenizer(question)['input_ids'])

            # 计算与所有公司的相似度
            similarities = [compute_similarity(question_tokens, c_tokens) for c_tokens in company_name_tokens]

            # 找到最高匹配度的公司名称
            max_index = np.argmax(similarities)
            if similarities[max_index] > 0:
                matched_entity = company_names[max_index]
                matched_csv = company_csv_files[max_index]

        else:  # SQL问题不需要匹配公司，直接写入csv文件
            pass

        csvwriter.writerow([q_id, question, category, matched_entity, matched_csv])

print('A02_finished')
