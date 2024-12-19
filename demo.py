import soundfile as sf
import gradio as gr
from yuyinzhuan文字 import zhuanwenzi


# 假设这是你的语音转文字函数
def speech_to_text(audio):
    process_audio(audio)
    p=zhuanwenzi()

    return p



def process_audio(audio):
    sr, data = audio
    sf.write('output.wav', data, sr)
    # 这里我们不反转音频，而是直接返回原始音频数据
    return (sr, data)

# 定义输入音频的参数
input_audio = gr.Audio(
    sources=["microphone", "upload"],  # 允许用户上传音频文件或使用麦克风
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)

# 定义输出文本的参数
output_text = gr.Textbox(label="输出文本")  # 添加一个文本框来显示转录的文本

# 创建 Gradio 界面
demo = gr.Interface(
    fn=speech_to_text,
    inputs=input_audio,
    outputs=output_text,  # 输出文本
    live=True
)

if __name__ == "__main__":
     demo.launch()


