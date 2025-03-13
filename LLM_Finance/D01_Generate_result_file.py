import pandas as pd
import json
import jsonlines

# **Step 1: 读取 CSV 文件**
sql_file = "./data/05_FA_SQL.csv"
text_file = "./data/06_FA_Text.csv"
output_file = "./submit_result.jsonl"

# 读取 05_FA_SQL.csv
df_sql = pd.read_csv(sql_file, dtype={"问题id": str})  # 统一问题id为 str 以保证匹配
df_text = pd.read_csv(text_file, dtype={"问题id": str})  # 读取 06_FA_Text.csv

# 建立字典方便快速查找 `06_FA_Text.csv` 的答案
text_answers = {row["问题id"]: row["FA"] for _, row in df_text.iterrows()}

# **Step 2: 处理数据**
results = []

for _, row in df_sql.iterrows():
    question_id = row["问题id"]
    question = row["问题"]
    fa_sql = row["FA"]

    # **如果 FA 不为 "N_A"，直接使用**
    if fa_sql != "N_A":
        answer = fa_sql
    else:
        # **如果 FA 为 "N_A"，查找 `06_FA_Text.csv`**
        if question_id in text_answers:
            fa_text = text_answers[question_id]
            # **如果 FA 包含 "无法"，用问题本身作为答案**
            if "无法" in fa_text:
                answer = question
            else:
                answer = fa_text
        else:
            # **如果 `06_FA_Text.csv` 也没有找到，直接返回问题**
            answer = question

    # **Step 3: 生成 JSON 结构**
    result = {
        "id": int(question_id),
        "question": question,
        "answer": answer
    }
    results.append(result)

# **Step 4: 写入 JSONL 文件**
with jsonlines.open(output_file, mode='w') as writer:
    writer.write_all(results)

print(f"✅ `submit_result.jsonl` 生成完毕！共 {len(results)} 条记录。")
