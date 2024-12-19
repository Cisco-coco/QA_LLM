---
language:
- zh
tags:
- chatglm
- pytorch
- Text-Generation
license: apache-2.0
widget:
- text: |-
    介绍下北京
    答：
library_name: peft
pipeline_tag: text-generation
base_model: THUDM/chatglm-6b
inference: false
---

# Chinese QA LoRA Model
ChatGLM中文问答LoRA模型

`chatglm-6b-belle-zh-lora` evaluate test data：

The overall performance of chatglm-6b-belle-zh-lora on QA **test**:

|input_text|predict|
|:-- |:--- |
|介绍下北京|北京是中国的首都，是中国的政治、文化和经济中心。它位于中国的北部，拥有悠久的历史和文化遗产，如故宫、长城和天坛等。北京也是中国最著名的城市之一，有着丰富的美食、文化和娱乐活动。北京还有许多旅游景点，如颐和园、圆明园和天坛等。北京是一个充满活力和机会的城市，吸引了来自全国和世界各地的人们前来探索。|

在中文开放测试集中的表现优异，继承了两方面的优势：1）微调的底座是ChatGLM-6B模型，中文的表现优于LLAMA，2）微调使用的是高质量100万条中文ChatGPT指令Belle数据集，微调后的模型对话效果优于原始ChatGLM-6B。


## Usage

本项目开源在textgen项目：[textgen](https://github.com/shibing624/textgen)，可支持ChatGLM模型，通过如下命令调用：

Install package:
```shell
pip install -U textgen
```

```python
from textgen import ChatGlmModel
model = ChatGlmModel("chatglm", "THUDM/chatglm-6b", peft_name="shibing624/chatglm-6b-belle-zh-lora")
r = model.predict(["介绍下北京\n答："])
print(r) # ['北京是中国的首都，是中国的政治、文化和经济中心。...']
```

## Usage (HuggingFace Transformers)
Without [textgen](https://github.com/shibing624/textgen), you can use the model like this: 

First, you pass your input through the transformer model, then you get the generated sentence.

Install package:
```
pip install transformers 
```

```python
import sys
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

sys.path.append('..')

model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map='auto')
model = PeftModel.from_pretrained(model, "shibing624/chatglm-6b-belle-zh-lora")
model = model.half().cuda()  # fp16
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

sents = ['介绍下北京\n答：',]
for s in sents:
    response = model.chat(tokenizer, s, max_length=128, eos_token_id=tokenizer.eos_token_id)
    print(response)
```

output:
```shell
介绍下北京
北京是中国的首都，是中国的政治、文化和经济中心。它位于中国的北部，拥有悠久的历史和文化遗产，如故宫、长城和天坛等。北京也是中国最著名的城市之一，有着丰富的美食、文化和娱乐活动。北京还有许多旅游景点，如颐和园、圆明园和天坛等。北京是一个充满活力和机会的城市，吸引了来自全国和世界各地的人们前来探索。
```


模型文件组成：
```
chatglm-6b-belle-zh-lora
    ├── adapter_config.json
    └── adapter_model.bin
```


### 训练数据集

1. 50万条中文ChatGPT指令Belle数据集：[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
2. 100万条中文ChatGPT指令Belle数据集：[BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
3. 5万条英文ChatGPT指令Alpaca数据集：[50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)
4. 2万条中文ChatGPT指令Alpaca数据集：[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
5. 69万条中文指令Guanaco数据集(Belle50万条+Guanaco19万条)：[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)


如果需要训练ChatGLM模型，请参考[https://github.com/shibing624/textgen](https://github.com/shibing624/textgen)


## Citation

```latex
@software{textgen,
  author = {Xu Ming},
  title = {textgen: Implementation of language model finetune},
  year = {2021},
  url = {https://github.com/shibing624/textgen},
}
```