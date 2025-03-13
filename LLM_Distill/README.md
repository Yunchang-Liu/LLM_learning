## 基于大模型蒸馏技术驱动小模型推理增强实验


### 项目概述
本项目旨在研究大模型的蒸馏方法，通过实现黑盒和白盒蒸馏策略，将大模型的推理能力传递到小模型。以Qwen2.5系列（3B和0.5B）为实验对象，通过对比不同蒸馏方法，使得小模型具备了反思和思考能力，在谜题任务中得到了显著提升。

### 主要工作
1）蒸馏数据构造：基于RiddleSense谜题数据集，使用DeepSeek-R1蒸馏得到包含CoT的答案，通过LLM-as-Judge方法进行二次校验；
2）黑盒蒸馏：使用DeepSeek-R1构造出的推理数据集作为教师模型的输出，SFT学生模型；
3）白盒蒸馏：LoRA微调Qwen2.5-3B作为教师模型，分别实现前向KL散度、反向KL散度、偏向KL散度作为学生模型的优化目标进行训练。

### 成果
本研究探索了如何通过蒸馏将大模型的推理能力高效迁移到小模型。黑盒蒸馏让参数量仅有0.5B的小模型在RiddleSense上的准确率由7%提升到了24%；使用Forward KL的白盒蒸馏达到了30%；而使用Reverse KL则更进一步提升到了32%。

### 流程
1. 下载riddlesense数据集(关于puzzle的,单项选择题)
2. 调用DeepSeek-R1接口，将riddlesense转换为推理数据集，并整理成指令数据集instruction, input, output的格式。记为DATA:训练集1000条，测试集500条
3. 开始大模型蒸馏实验
    - **黑盒蒸馏**：首先使用DATA利用LoRA微调Qwen2.5-0.5B-Instruct，测试其在推理数据集上的性能。记为Blackbox-Distill-Qwen2.5-0.5B:ACC=xx.xx%
    - **白盒蒸馏**：然后使用DATA利用LoRA微调Qwen2.5-3B-Instruct，再通过模型蒸馏方法对齐**没有经过微调的**Qwen2.5-0.5B-Instruct，测试其在推理数据集上的性能。记为Whitebox-Distill-Qwen2.5-0.5B:ACC=xx.xx%


### 文件介绍
- data_process1-3.py用于处理riddlesense数据集，将其转换为推理数据集，并整理成指令数据集instruction, input, output的格式
- fine_tune_3B.py用于LoRA微调大模型
- train.py用于训练蒸馏模型
- utils.py定义了不同的KL散度计算方法
- inference_and_evaluate.py用于推理和评估模型在puzzle数据集上的性能


### 实验结果
| Model                               | Loss Function                     | Accuracy (ACC) |
|-------------------------------------|-----------------------------------|----------------|
| Qwen2.5-3B-Instruct-LoRA            | -                                 | 57.0%          |
| Qwen2.5-0.5B-Instruct               | -                                 | 7.0%           |
| Blackbox-Distill-Qwen2.5-0.5B       | -                                 | 26.0%          |
| Whitebox-Distill-Qwen2.5-0.5B       | Forward KL                        | 30.0%          |
| Whitebox-Distill-Qwen2.5-0.5B       | Reverse KL                        | 31.0%          |
