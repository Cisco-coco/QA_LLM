import gradio as gr
import pyttsx3

# 文字转语音函数
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    # 由于 pyttsx3 直接播放语音，我们返回一个消息表示播放完成
    # return "语音播放完成"

# 定义输入文本的参数
input_text = gr.Textbox(label="输入文本")

# 创建 Gradio 界面
demo = gr.Interface(
    fn=text_to_speech,
    inputs=input_text,
    outputs=None,  # 输出文本消息
    live=True
    #如果是True，就是自动转为语音，如果是False，需要你点击submit转为语音。
    # 第一种适合文本已经确定，第二种适合文本自己输入。
    # 在我们模型中，因为response已经确定所以适合第一种，但第二种是给用户一个选择听不听语音的权限。
)

if __name__ == "__main__":
    demo.launch()