from transformers import Trainer, HfArgumentParser
from dataclasses import dataclass, field
@dataclass
class DataArguments:
    memory_search_top_k: int = field(default=2)
    memory_basic_dir: str = field(default='../../memories/')
    memory_file: str = field(default='update_memory_0512_eng.json')
    language: str = field(default='cn')
    max_history: int = field(default=7,metadata={"help": "maximum number for keeping current history"},)
    enable_forget_mechanism: bool = field(default=False)
@dataclass
class ModelArguments:
    model_type: str = field(
        default="chatglm",
        metadata={"help": "model type: chatglm / belle"},
    )
    base_model: str = field(
        default="/data/sshfs/dataroot/models/THUDM/chatglm-6b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    adapter_model: str = field(
        # default="/data/sshfs/94code/MemoryBank-SiliconFriend/model/ChatGLM-LoRA-checkpoint",
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