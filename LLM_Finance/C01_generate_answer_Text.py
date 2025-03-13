import os
import re
import math
import csv
import pandas as pd
from collections import defaultdict
import thulac
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modelscope import AutoModelForCausalLM, AutoTokenizer


# **Step 1: 配置环境**
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置 GPU 设备

# **Step 2: 参数定义**
txt_dir = "./bs_challenge_financial_14b_dataset/pdf_txt_file"  # 存放所有 txt 文档
csv_file = "./data/02_question_to_entity.csv"  # 问题 CSV 文件
output_file = "./data/06_FA_Text.csv"  # 结果输出文件
model_dir = "./Tongyi-Finance-14B-Chat-Int4"
n = 10  # 取最相关的 n 个文本片段

# **Step 3: 读取问题 CSV**
q_file = pd.read_csv(csv_file, delimiter=",", header=0)

# **Step 4: 加载 LLM**
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

# **Step 5: 加载中文分词工具**
thu = thulac.thulac(seg_only=True)
stop_words = set(open('./data/stopwords.txt').read().split("\n"))

# **Step 6: 计算 TF-IDF 相关性**
def extrac_doc_page(thu, stop_words, doc, question):
    """ 使用 TF-IDF 计算文本相关性 """
    seg_list = thu.cut(question)  # 中文分词
    seg_list = [x[0] for x in seg_list]
    key_words = list(set(seg_list) - stop_words)  # 去掉停用词

    key_word_tf = defaultdict(int)  # 解决 KeyError
    key_word_idf = defaultdict(list)  # 解决 KeyError

    for idx, p in enumerate(doc):
        finds = re.findall('|'.join(key_words), p)  # 在文本中查找关键词
        for r in finds:
            key_word_tf[r] += 1  # 词频递增
            key_word_idf[r].append(idx)  # 记录该词在哪些段落出现

    key_tfidf = []
    for key in key_words:
        tf = 1  # 这里 TF 设为 1，避免被零除
        idf = math.log(len(doc) / (len(set(key_word_idf[key])) + 1))  # 计算 IDF
        key_tfidf.append((key, tf * idf))

    key_tfidf.sort(key=lambda x: x[1], reverse=True)

    doc_p = [[i, 0] for i in range(len(doc))]
    for key in key_tfidf:
        p_id = list(set(key_word_idf[key[0]]))
        for p in p_id:
            doc_p[p][1] += key[1]

    doc_p.sort(key=lambda x: x[1], reverse=True)
    after_rank_doc = [doc[i[0]] for i in doc_p[:n]]
    
    return "\n\n".join(after_rank_doc), key_tfidf, doc_p

# **Step 7: LangChain 文本切分器**
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["。", "，"]
)

# **Step 8: 处理问题 & 逐行写入 CSV**
# 读取已完成的问题 ID
completed_questions = set()
if os.path.exists(output_file):
    df_completed = pd.read_csv(output_file)
    completed_questions = set(df_completed["问题id"].tolist())

with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    if len(completed_questions) == 0:
        csvwriter.writerow(["问题id", "问题", "FA"])  # 写入表头

    for _, row in tqdm(q_file.iterrows(), total=len(q_file)):
        question_id = row["问题id"]
        # 跳过已经完成的问题
        if question_id in completed_questions:
            print(f"✅ 问题 {question_id} 答案已存在")
            continue

        question = row["问题"]
        category = row["分类"]
        company_name = row["对应实体"]
        txt_filename = row["csv文件名"]

        # 仅处理 `Text` 类问题
        if category != "Text":
            continue

        txt_filename = txt_filename.replace(".PDF.csv", ".txt")
        txt_path = os.path.join(txt_dir, txt_filename)

        if not os.path.exists(txt_path):
            print(f"❌ 找不到 {txt_path}")
            csvwriter.writerow([question_id, question, "无法回答"])  # 文件不存在，写入默认答案
            continue

        # **Step 9: 读取 & 处理 txt 文本**
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().replace("\n", "")

        # 使用 LangChain 进行文本切分
        documents = text_splitter.split_text(text)

        # 计算 TF-IDF 相关性
        retrieved_text, _, _ = extrac_doc_page(thu, stop_words, documents, question)

        # **Step 10: 调用 LLM 生成答案**
        prompt = f"""
        你是一个能精准提取文本信息并回答问题的AI。请你用下面提供的材料回答问题，材料是：
        ```{retrieved_text[:5000]}```。
        
        请根据以上材料回答问题：" {question} "。
        如果能根据给定材料回答，则提取出最合理的答案来回答问题，并回答完整内容。
        如果不能找到答案，则回答 "无法回答"。
        """

        response, _ = model.chat(tokenizer, prompt, history=None, max_new_tokens=512)

        # **Step 11: 立即写入 CSV**
        csvwriter.writerow([question_id, question, response])
        print(f"✅ 问题 {question_id} 处理完成，已写入 CSV")

print("✅ `06_FA_Text.csv` 生成完毕！")
