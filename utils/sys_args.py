from transformers import Trainer, HfArgumentParser
from dataclasses import dataclass, field
@dataclass
class DataArguments:
    memory_search_top_k: int = field(default=2)
    memory_basic_dir: str = field(default='../../memories/') # memory在磁盘上的存储目录
    memory_file: str = field(default='update_memory_0512_eng.json') # memory存储文件名
    language: str = field(default='cn')
    max_history: int = field(default=7,metadata={"help": "maximum number for keeping current history"},)
    enable_forget_mechanism: bool = field(default=False) # 是否遗忘记忆 （做小规模测试的话启不启用对交互效果没啥影响）
@dataclass
class ModelArguments:
    model_type: str = field(
        default="chatglm",
        metadata={"help": "model type: chatglm / belle"},
    )
    base_model: str = field(
        default="/data/sshfs/dataroot/models/THUDM/chatglm-6b", # 源模型地址（修改成你自己的）
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    adapter_model: str = field(
        # default="/data/sshfs/94code/MemoryBank-SiliconFriend/model/ChatGLM-LoRA-checkpoint", # LoRA模型地址（修改成你自己的） 
        default="/data/sshfs/94code/MemoryBank-SiliconFriend/model/shibing624/chatglm-6b-belle-zh-lora",
        metadata={"help": "Path to lora adapter model"},
    )
    ptuning_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to pretrained prefix embedding of ptuning"},
    )
    

    # prompt_column
    # train_file: str = field(default="/home/t-qiga/azurewanjun/SiliconGirlfriend/data/merge_data/only_mental_0426.json")

data_args,model_args = HfArgumentParser(
    (DataArguments,ModelArguments)
).parse_args_into_dataclasses()