import jsonlines
import csv

def read_jsonl(path):
    """ 读取JSONL文件,返回列表,每个元素是一个字典 """
    with jsonlines.open(path, "r") as json_file:
        return [obj for obj in json_file.iter(type=dict, skip_invalid=True)]

# 读取JSONL文件
questions = read_jsonl('./bs_challenge_financial_14b_dataset/question.json')

# 写入CSV文件
# 两列：id是int类型, Question是str类型
csv_path = './data/00_all_questions.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['问题id', '问题'])  # 写入表头
    for q in questions:
        q_id = q['id']
        q_text = q['question'].replace(' ', '').replace(',', '，')
        csvwriter.writerow([q_id, q_text])  # 写入数据


print(f"✅ CSV 文件已保存至 {csv_path}")
