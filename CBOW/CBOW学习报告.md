# **词袋模型相关矩阵学习与应用**
（包含较多矩阵公式，vscode安装 Markdown Preview Enhanced插件可预览）

## **一、 词袋模型简介**

词袋模型（Bag of Words, BoW）是自然语言处理（NLP）中最基础的文本特征提取方法之一。它将文本视为一个**单词集合**，忽略单词的顺序，仅统计各个单词在文本中出现的频率或权重。词袋模型通过词频构建向量空间，用于后续的文本分类、相似度计算等任务。

词袋模型的优势在于实现简单且容易理解，但缺点也很明显：
- **忽略单词顺序**，无法捕捉上下文语义。
- **语义缺失**：相似但不同单词的关系无法体现。
- **词汇量大**：需要构建一个词汇表，词汇量增大时计算复杂度较高。

通过结合TF-IDF（词频-逆文档频率）方法，词袋模型可以进一步提升效果，解决一些高频词对结果影响较大的问题。

---

### **1.1 词袋模型（BOW）矩阵运算分析**

在词袋模型中，矩阵运算的核心体现在从输入文本到特征向量化的过程，以及在通过嵌入矩阵计算词向量。以下详细解析了其运算逻辑：

#### **1.1.1. 基本词袋模型运算**

1. **特征向量构建**：
   - 假设文本集合为 \( D = \{d_1, d_2, \dots, d_n\} \)，词汇表为 \( V = \{w_1, w_2, \dots, w_m\} \)，其中 \( n \) 是文档数，\( m \) 是词汇数。
   - 每个文档 \( d_i \) 会被表示为一个长度为 \( m \) 的向量 \( v_i \)，其中 \( v_{ij} \) 表示单词 \( w_j \) 在文档 \( d_i \) 中的词频。

2. **TF-IDF 计算**：
   - **TF（词频）**：表示单词在文档中的出现频率。
   - **IDF（逆文档频率）**：衡量单词在整个文档集合中的重要程度。计算公式为：
     \[
     \text{IDF}(w) = \log\frac{N}{n_w + 1}
     \]
     其中 \( N \) 是总文档数，\( n_w \) 是包含单词 \( w \) 的文档数。
   - TF-IDF 特征向量表示为：
     \[
     \text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)
     \]

3. **余弦相似度**：
   - 两个向量之间的相似度通过余弦相似度计算，公式如下：
     \[
     \text{cosine\_similarity}(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||}
     \]
   - 余弦相似度用于判断两个文本的相似程度。

---

#### **1.1.2. 连续词袋模型（continuous bag of words，CBOW）模型的矩阵运算**

CBOW 模型进一步利用上下文词预测目标词，学习每个单词的**词向量表示**，涉及的矩阵运算包括以下几个步骤：

**1 输入层（One-Hot 编码）**
- 输入的上下文词通过 One-Hot 编码表示。
- 假设词汇表大小为 \( V \)，上下文词数量为 \( C \)：
  - 单个上下文词的 One-Hot 向量 \( x_i \) 为：
    \[
    x_i = [0, 0, 1, 0, \dots, 0] \quad \text{（长度为 \( V \)）}
    \]
  - 所有上下文词组成矩阵 \( X \)：
    \[
    X = \begin{bmatrix}
    0 & 0 & 1 & 0 & \dots & 0 \\
    0 & 1 & 0 & 0 & \dots & 0 \\
    \dots & \dots & \dots & \dots & \dots & \dots
    \end{bmatrix} \quad \text{（维度 \( C \times V \)）}
    \]


**2 嵌入矩阵映射（Embedding Matrix）**
- 使用嵌入矩阵 \( W_{in} \) （维度为 \( V \times d \)，\( d \) 是嵌入向量的维度）将 One-Hot 向量投影到低维空间：
  \[
  h_i = x_i \cdot W_{in}
  \]
- 上下文词的嵌入表示矩阵为：
  \[
  H = X \cdot W_{in} \quad \text{（维度 \( C \times d \)）}
  \]
- 本质上，矩阵乘法是从 \( W_{in} \) 中提取每个上下文词对应的嵌入向量。

**3 上下文向量平均**
- 将上下文词的嵌入向量取平均，得到隐藏层向量 \( h \)：
  \[
  h = \frac{1}{C} \sum_{i=1}^{C} h_i
  \]
- \( h \) 是一个长度为 \( d \) 的向量，表示上下文的综合信息。

**2.4 输出层（预测目标词）**
- 使用输出层权重矩阵 \( W_{out} \) （维度为 \( d \times V \)）将隐藏层向量映射回词汇表维度：
  \[
  z = h \cdot W_{out} \quad \text{（维度 \( 1 \times V \)）}
  \]
- 对 \( z \) 应用 Softmax 函数，计算目标词的预测概率分布：
  \[
  p(y=j|h) = \frac{\exp(z_j)}{\sum_{k=1}^V \exp(z_k)}
  \]
- \( p(y=j|h) \) 表示预测目标词为 \( j \) 的概率。
  
**5. 模型训练**

通过反向传播优化嵌入矩阵 \( W_{in} \) 和 \( W_{out} \)，逐渐学习到高质量的词向量。

目标函数：
   - 使用交叉熵损失函数衡量预测分布和真实目标的差异：
     \[
     \mathcal{L} = - \log p(y=y_{true}|h)
     \]
     其中 \( y_{true} \) 是目标词的真实索引。

梯度更新：
   - 使用优化器（如 SGD）根据梯度更新模型参数：
     \[
     W \gets W - \eta \cdot \nabla_W \mathcal{L}
     \]
     - \( W \)：表示嵌入矩阵 \( W_{in} \) 或输出矩阵 \( W_{out} \)。
     - \( \eta \)：学习率。



**6. 最终获得词向量**
经过多轮迭代训练后，隐藏层的嵌入矩阵 \( W_{in} \) 即为每个单词的**词向量表示**。

- 提取词向量：
  - 对于单词 \( w \)，其索引为 \( i \)，词向量为：
    \[
    \text{词向量} = W_{in}[i]
    \]
- 这些词向量可以用来表示单词的语义关系，并在下游任务中（如文本分类、相似度计算）发挥重要作用。


---

#### **1.1.3 CBOW程序实现**

见附件CBOW.py文件
---
可以读取本地utf-8文档训练，最后注释掉的内容可以生成本地的词向量缓存。


基于 PyTorch 的 CBOW 实现流程：
1. **数据预处理**：分词、构建词汇表、生成上下文-目标词训练数据。
2. **模型定义**：使用 `torch.nn` 构建输入到隐藏层，再映射到输出层的网络。
3. **训练**：基于交叉熵损失函数和 SGD 优化器，迭代优化模型权重。
4. **保存**：将训练后的 embedding 矩阵保存供后续任务使用。


---

## **二、利用词袋语言模型实现Q-A系统**

### **2.1 Q-A系统结构分析**

基于词袋模型的问答系统（Q-A系统）采用了 **TF-IDF** 和余弦相似度来匹配用户查询与知识库中的答案。系统的核心模块包括：

1. **数据预处理**：
   - 使用中文分词工具（如 `jieba`）对文本进行分词。
   - 构建知识库的词袋模型特征。

2. **文本向量化**：
   - 使用 `TfidfVectorizer` 将知识库和用户查询文本转换为 TF-IDF 特征向量。

3. **相似度计算**：
   - 通过余弦相似度计算用户查询与知识库中各条目之间的相似度，选取最相似的文本作为回答。

4. **系统接口**：
   - 使用 Gradio 实现用户输入查询，系统返回最匹配的答案。

---

### **2.2 Python程序实现**

见附件QA.py文件
由于CBOW做不了生成式语言模型，故实现结果较差，大致表现为上传utf-8的txt文档后，进行提问，程序从文档中选择词向量相近的句子，故回答毫无逻辑依赖于上传的文档。
要求gradio安装3.36.1版本，否则会报错。
