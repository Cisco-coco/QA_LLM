import jieba  # 中文分词库
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import re
from tqdm import tqdm  # 导入 tqdm 库
from safetensors.torch import save_file  # 导入 safetensors 保存函数
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers  # 使用tokenizers库生成tokenizer.json

# 读取书籍内容（假设书籍内容保存在 merged.txt 中）
with open("merged.txt", "r", encoding="utf-8") as f:
    book_text = f.read()

# 中文分词函数，去除特殊字符
def chinese_cut(text):
    # 通过正则表达式去除标点符号和其他特殊字符
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    return " ".join(jieba.cut(text))

# 将整个书本内容按句子分割为列表，假设每行一个句子
sentences = book_text.split("\n")

# 对所有中文句子进行分词
segmented_sentences = [chinese_cut(sentence) for sentence in sentences]

# 使用 CountVectorizer 来构建词袋模型
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(segmented_sentences)

# 获取词汇表
vocab = vectorizer.get_feature_names_out()

# 在词汇表中加入 <UNK>，并更新词汇表
vocab = list(vocab) + ["<UNK>"]  # 添加 <UNK> 标记
word_to_idx = {word: idx for idx, word in enumerate(vocab)}  # 词汇 -> 索引
idx_to_word = {idx: word for idx, word in enumerate(vocab)}  # 索引 -> 词汇

# 词汇表大小
voc_size = len(vocab)
print("词汇表大小：", voc_size)

# 保存词汇表到 vocab.txt
with open("vocab.txt", "w", encoding="utf-8") as f:
    for word in vocab:
        f.write(f"{word}\n")

# 保存 tokenizer 的配置到 tokenizer_config.json
tokenizer_config = {
    "vocab_size": voc_size,
    "tokenizer_class": "jieba"
}
with open("tokenizer_config.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, indent=4)

# 生成 CBOW 训练数据
def create_cbow_dataset(sentences, window_size=2):
    data = []
    for sentence in sentences:
        sentence = sentence.split()  # 将句子分割成单词列表
        for idx, word in enumerate(sentence):
            context_words = sentence[max(idx - window_size, 0):idx] + sentence[idx + 1:min(idx + window_size + 1, len(sentence))]
            # 如果上下文词为空，则跳过
            if len(context_words) == 0:
                continue
            data.append((word, context_words))
    return data

# 使用函数创建 CBOW 训练数据
cbow_data = create_cbow_dataset(segmented_sentences)
print("CBOW 数据样例（未编码）：", cbow_data[:3])

# 定义 CBOW 模型
class CBOW(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(CBOW, self).__init__()
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)
    
    def forward(self, X):
        embeddings = self.input_to_hidden(X)  # [num_context_words, embedding_size]
        hidden_layer = torch.mean(embeddings, dim=0)  # [embedding_size]
        output_layer = self.hidden_to_output(hidden_layer.unsqueeze(0))  # [1, voc_size]
        return output_layer

# 超参数
learning_rate = 0.001
epochs = 1
embedding_size = 100

# 选择设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将模型转移到正确的设备
cbow_model = CBOW(voc_size, embedding_size).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cbow_model.parameters(), lr=learning_rate)

# 用于记录每轮的损失
loss_values = []

# One-Hot 编码函数，增加错误字符检查
def one_hot_encoding(word, word_to_idx):
    # 如果词汇表中不存在该词，则使用 <UNK> 替代
    if word not in word_to_idx:
        word = "<UNK>"
    one_hot = torch.zeros(len(word_to_idx))
    one_hot[word_to_idx[word]] = 1
    return one_hot

# 训练过程
for epoch in range(epochs):
    loss_sum = 0  # 初始化损失总和
    # 使用 tqdm 包装训练数据，用于显示进度条
    for target, context_words in tqdm(cbow_data, desc=f"Epoch {epoch+1}/{epochs}", ncols=100, ascii=True):
        # 将上下文词转换为 One-Hot 向量并堆叠（转移到 GPU 或 CPU）
        X = torch.stack([one_hot_encoding(word, word_to_idx) for word in context_words]).float().to(device)
        # 将目标词转换为索引值（转移到 GPU 或 CPU）
        y_true = torch.tensor([word_to_idx.get(target, word_to_idx["<UNK>"])], dtype=torch.long).to(device)
        
        # 前向传播
        y_pred = cbow_model(X)  # 计算预测值
        
        # 计算损失
        loss = criterion(y_pred, y_true)
        loss_sum += loss.item()
        
        # 反向传播并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss_sum / len(cbow_data)}")
        loss_values.append(loss_sum / len(cbow_data))

# 绘制训练损失曲线
plt.plot(range(1, epochs // 100 + 1), loss_values)
plt.title('训练损失曲线')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.show()

# # 保存模型
# torch.save(cbow_model.state_dict(), "pytorch_model.bin")

# # 如果使用 safetensors 库保存 safetensors 格式
# save_file(cbow_model.state_dict(), "model.safetensors")

# # 保存 eval_results.txt
# eval_results = {"final_loss": loss_sum / len(cbow_data)}
# with open("eval_results.txt", "w", encoding="utf-8") as f:
#     f.write(f"Final Loss: {eval_results['final_loss']}\n")

# # 生成 tokenizer.json 文件
# tokenizer = Tokenizer(models.BPE())  # 使用 BPE 模型
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# tokenizer.decoder = decoders.ByteLevel()
# trainer = trainers.BpeTrainer(vocab_size=voc_size, special_tokens=["<UNK>"])
# tokenizer.train_from_iterator(segmented_sentences, trainer)

# # 保存 tokenizer.json 文件
# tokenizer.save("tokenizer.json")

# # 生成 special_tokens_map.json
# special_tokens_map = {
#     "unk_token": "<UNK>"
# }
# with open("special_tokens_map.json", "w", encoding="utf-8") as f:
#     json.dump(special_tokens_map, f, indent=4)

# # 生成 config.json 文件
# config = {
#     "model_type": "CBOW",
#     "vocab_size": voc_size,
#     "embedding_size": embedding_size
# }
# with open("config.json", "w", encoding="utf-8") as f:
#     json.dump(config, f, indent=4)
