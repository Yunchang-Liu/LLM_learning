import csv
import sqlite3
import pandas as pd
from tqdm import tqdm


# 数据库连接
conn = sqlite3.connect('./bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db')
cs = conn.cursor()

# 可能相互替换的表名
similar_tables = ['基金股票持仓明细', '基金债券持仓明细', '基金可转债持仓明细']

# 需要替换的内容
replacement_dict = {
    "B股票日行情表": "A股票日行情表",
    "创业板日行情表": "A股票日行情表",
    " 股票日行情表": " A股票日行情表",
    " 港股日行情表": " 港股票日行情表",
    "”": "",
    "“": ""
}

new_question_file = pd.read_csv('./data/03_question_SQL.csv', delimiter=",", header=0)
output_file = './data/04_question_SQL_exed.csv'

with open(output_file, 'w', newline='', encoding='utf-8-sig') as g:
    csvwriter = csv.writer(g)
    csvwriter.writerow(['问题id', '问题', 'SQL语句', '能否成功执行', '执行结果'])

    for _, row in tqdm(new_question_file.iterrows(), total=len(new_question_file), desc="Executing SQL"):
        q_id = row['问题id']
        question = row['问题']
        sql_query = row['SQL语句']

        # 预处理 SQL 语句
        for old, new in replacement_dict.items():
            sql_query = sql_query.replace(old, new)

        execution_success = 0  # 0: 失败，1: 原SQL成功，2: 替换后成功
        execution_result = "N_A"
        replaced_table = "N_A"

        # 尝试执行SQL
        try:
            cs.execute(sql_query)
            execution_result = str(cs.fetchall())
            execution_success = 1  # 原始SQL执行成功
        except:
            # 检查是否包含需要替换的表
            for table in similar_tables:
                if table in sql_query:
                    replaced_table = table
                    break
            
            # 如果发现匹配的表，尝试替换
            if replaced_table != "N_A":
                for alternative_table in similar_tables:
                    modified_sql = sql_query.replace(replaced_table, alternative_table)
                    try:  # 尝试执行替换后的SQL
                        cs.execute(modified_sql)
                        execution_result = str(cs.fetchall())
                        execution_success = 2  # 替换表名后成功执行
                        break
                    except:
                        pass  # 如果仍然失败，继续尝试其他替换

        # 写入结果到 CSV
        csvwriter.writerow([q_id, question, sql_query, execution_success, execution_result])

# 关闭数据库连接
cs.close()
conn.close()

print("Execution completed ✅")