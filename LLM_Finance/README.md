# **大模型智能问答系统**

本项目结合RAG-ICL+指令精调，实现了支持NL2SQL、文本理解两方面问题的大模型智能问答系统。

---

## **1. 项目结构**

```bash
.
├── A00_question_to_csv.py         # 解析原始问题数据并转换为 CSV
├── A01_question_classify.py       # 识别问题类别 (SQL 或 Text)
├── A02_question_to_entity.py      # 确定 Text 类问题的相关公司名称
├── B01_generate_SQL.py            # 生成 SQL 查询
├── B02_apply_SQL.py               # 执行 SQL 查询并获取结果
├── B03_generate_answer_SQL.py     # 根据 SQL 结果生成最终答案
├── C01_generate_answer_Text.py    # 处理 Text 问题并用 LLM 生成答案
├── D01_Generate_result_file.py    # 整合所有答案并输出 JSONL 文件
├── data                           # 存放中间数据文件
│   ├── 05_FA_SQL.csv              # SQL 问题的答案
│   ├── 06_FA_Text.csv             # Text 问题的答案
│   └── submit_result.jsonl        # 最终提交结果
└── README.md                      # 项目说明
```
---

## **2. 处理流程**

### **Step 1: 数据预处理**
- **A00_question_to_csv.py**  
  - 解析原始问题数据并转换为 `CSV` 格式，方便后续处理。
- **A01_question_classify.py**  
  - 识别问题类别 (SQL 或 Text)。
- **A02_question_to_entity.py**  
  - 如果问题类别为 Text，则匹配问题相关的公司名称及其文档。

---

### **Step 2: 处理 SQL 类问题**
- **B01_generate_SQL.py**  
  - 解析 SQL 类问题并生成 SQL 查询语句。
- **B02_apply_SQL.py**  
  - 执行 SQL 查询，获取结果，并存入 `data/04_question_SQL_exed.csv`。
- **B03_generate_answer_SQL.py**  
  - 读取 SQL 查询结果，并使用大模型对 SQL 结果进行解析、整理，输出 `05_FA_SQL.csv`。

---

### **Step 3: 处理 Text 类问题**
- **C01_generate_answer_Text.py**  
  - 读取 Text 类问题对应的 `txt` 文件，使用 **TF-IDF 相关性计算** 从文档中检索最相关的 10 个片段。
  - 使用 **LangChain** 进行文本切分，提高检索效果。
  - 通过 **通义千问大模型 (Tongyi-Finance-14B-Chat-Int4)** 生成答案，并存入 `06_FA_Text.csv`。

---

### **Step 4: 生成最终 JSONL 文件**
- **D01_Generate_result_file.py**  
  - 读取 `05_FA_SQL.csv` 和 `06_FA_Text.csv`，整合 SQL 和 Text 问题的答案：
    - 如果 `FA` 字段不为 `"N_A"`，则直接使用 SQL 生成的答案。
    - 如果 `FA == "N_A"`，则去 `06_FA_Text.csv` 查找对应 `问题id` 的答案。
    - 如果答案中含 `"无法"`，则返回问题本身作为答案。
  - 生成 `submit_result.jsonl` 作为最终输出。

---

## **3. 关键技术**

### **TF-IDF 文本相关性计算**
TF-IDF (Term Frequency - Inverse Document Frequency) 用于衡量文本中某个词对文档的重要性。

- **TF (词频)**:
  - TF(t) = 该词在文档中出现的次数 / 文档中的总词数

- **IDF (逆文档频率)**:
  - IDF(t) = log(总文档数 / (包含该词的文档数 + 1))

- **TF-IDF 总分数**:
  - 所有关键词的 TF-IDF 之和

我们对每个问题的关键词计算 TF-IDF 分数，选取得分最高的 10 个文本片段作为 LLM 的输入。


---

## **4. 运行方法**

### **(1) 运行 SQL 处理流程**
```bash
python A00_question_to_csv.py
python A01_question_classify.py
python A02_question_to_entity.py
python B01_generate_SQL.py
python B02_apply_SQL.py
python B03_generate_answer_SQL.py
```

### **(2) 运行 Text 处理流程**
```bash
python C01_generate_answer_Text.py
```

### **(3) 生成最终 JSONL 结果**
```bash
python D01_Generate_result_file.py
```
