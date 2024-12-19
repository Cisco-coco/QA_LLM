import jieba
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 自定义分词器：使用 jieba 进行分词
def jieba_tokenizer(text):
    return jieba.lcut(text)

# 词袋模型：使用TF-IDF计算文本相似度
class BagOfWordsQA:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, stop_words=None)  # 设置分词器
        self.knowledge_base = []

    def load_knowledge_base(self, knowledge_base):
        if not knowledge_base:
            raise ValueError("The knowledge base is empty, cannot train the TF-IDF model.")
        # 对知识库的每个句子进行分词并训练
        self.knowledge_base = knowledge_base
        self.vectorizer.fit(self.knowledge_base)
        print("Knowledge base loaded successfully!")

    def answer_query(self, query):
        if not self.knowledge_base:
            return "Knowledge base is empty, please upload a valid knowledge base."
        try:
            # 分词后将查询转换为向量
            query_vector = self.vectorizer.transform([query])
            knowledge_vectors = self.vectorizer.transform(self.knowledge_base)
            similarities = cosine_similarity(query_vector, knowledge_vectors).flatten()

            print("Similarities:", similarities)  # 调试输出

            # 添加相似度阈值判断
            max_similarity = similarities.max()
            if max_similarity < 0.1:
                return "No relevant answer found."

            best_match_index = similarities.argmax()
            return f"Answer: {self.knowledge_base[best_match_index]} (Score: {max_similarity:.2f})"
        except Exception as e:
            return f"Error processing query: {str(e)}"

qa_system = BagOfWordsQA()

# 文件上传处理函数
def file_upload(file):
    try:
        with open(file.name, "r", encoding="utf-8") as f:
            knowledge_base = [line.strip() for line in f.readlines() if line.strip()]
        if not knowledge_base:
            return "The file is empty. Please upload a valid knowledge base."
        qa_system.load_knowledge_base(knowledge_base)
        return f"Knowledge base loaded successfully! Total entries: {len(knowledge_base)}"
    except Exception as e:
        return f"Error loading file: {str(e)}"

# 问答处理函数
def main(query):
    return qa_system.answer_query(query)

# Gradio界面实现
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Bag of Words QA System with Chinese Support")
        file_upload_output = gr.Textbox(label="File Upload Status", interactive=False)
        file_upload_button = gr.File(label="Upload Knowledge Base (txt)", file_types=[".txt"])
        file_upload_button.change(file_upload, inputs=file_upload_button, outputs=file_upload_output)

        user_input = gr.Textbox(label="Ask a question", placeholder="Enter your question here...")
        submit_button = gr.Button("Submit")
        chatbot_output = gr.Textbox(label="Answer", interactive=False)

        submit_button.click(main, inputs=user_input, outputs=chatbot_output)

    demo.launch(share=True)

if __name__ == "__main__":
    gradio_interface()
