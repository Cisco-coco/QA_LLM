# -*- coding:utf-8 -*-

import os, shutil

import logging
import sys
import time, platform
from app_modules.utils import *
#  
from app_modules.presets import *
from app_modules.overwrites import *

import signal,json
import gradio as gr
import nltk

import torch
from transformers.generation.utils import LogitsProcessorList
prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../memory_bank')
print(bank_path)
sys.path.append(prompt_path)
sys.path.append(bank_path)
from utils.prompt_utils import *
from utils.memory_utils import enter_name, summarize_memory_event_personality, save_local_memory
from utils.model_utils import load_chatglm_tokenizer_and_model,load_lora_chatglm_tokenizer_and_model,load_prefix_chatglm_tokenizer_and_model, InvalidScoreLogitsProcessor
from utils.sys_args import data_args,model_args
nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

# modification
# sys.path.append('/home/t-qiga/azurewanjun/SiliconGirlfriend/code/SiliconLover/BELLE-based/memory_bank/')
current_path = os.path.dirname(os.path.abspath(__file__))

 
from summarize_memory import summarize_memory
from memory_retrieval.configs.model_config import *
# from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
if not os.path.exists(memory_dir):
    json.dump({},open(memory_dir,"w",encoding="utf-8"))

language = data_args.language
if data_args.enable_forget_mechanism:
    from memory_retrieval.forget_memory import LocalMemoryRetrieval
else:
    from memory_retrieval.local_doc_qa import LocalMemoryRetrieval

local_memory_qa = LocalMemoryRetrieval()
EMBEDDING_MODEL = EMBEDDING_MODEL_CN if language == 'cn' else EMBEDDING_MODEL_EN
local_memory_qa.init_cfg(
                        embedding_model=EMBEDDING_MODEL,
                        embedding_device=EMBEDDING_DEVICE,
                        top_k=data_args.memory_search_top_k,
                        language=language)

meta_prompt = generate_meta_prompt_dict_chatglm_app()[language]
new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatglm()[language]
user_keyword = generate_user_keyword()[language]
ai_keyword = generate_ai_keyword()[language]
boot_name = boot_name_dict[language]
boot_actual_name = boot_actual_name_dict[language]


# tokenizer, model= load_prefix_chatglm_tokenizer_and_model(base_model,adapter_model)
tokenizer, model= load_lora_chatglm_tokenizer_and_model(model_args.base_model,model_args.adapter_model)


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)



def chat(model, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
            do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, 
            user_memory=None,
            user_name=None,
            user_memory_index=None,
            **kwargs):
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = { "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor,**kwargs}

    prompt = build_prompt_with_search_memory_chatglm_app(history,query,user_memory,user_name,user_memory_index,local_memory_qa,meta_prompt,new_user_meta_prompt,user_keyword,ai_keyword,boot_actual_name,language)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    response = model.process_response(response)
    response = clean_result(response,prompt,stop_words=[user_keyword])
    # history = history + [(query, response)]
    return response

 
def clean_result(result,prompt,stop_words):
    result = result.replace(prompt,"").strip() 
    # if is_stop_word_or_prefix(result, stop_words) is False:
    for stop in stop_words:
        if stop in result:
            result = result[:result.index(stop)].strip()
    result = result.strip(" ")
    return convert_to_markdown(result)


def predict_new(
    text,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
    user_name,
    user_memory,
    user_memory_index
):
    if text == "":
        return history, history, "Empty context."
        
   
    if len(history) > data_args.max_history:
        history = history[-data_args.max_history:]
    # print(history)
    response = chat(model,tokenizer,text,history=history,
                    num_beams=1,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=1,
                    max_length=max_context_length_tokens,
                    max_new_tokens=max_length_tokens,
                    user_memory=user_memory,
                    user_name=user_name,
                    user_memory_index=user_memory_index)
    result = response

    torch.cuda.empty_cache()
   
    a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
                    [text, convert_to_markdown(result)]], history + [[text, result]]
    # a, b = [[y[0], convert_to_markdown(y[1])] for y in history] ,history 
    if user_name:
        save_local_memory(memory,b,user_name,data_args)
    
    return a, b, "Generating..."
    
    

def main():
    history = []
    global stop_stream
    global memory
    memory = json.loads(open(memory_dir,"r",encoding="utf-8").read())
    print('Please Enter Your Name:')
    user_name = input("\n用户名：")
    print(memory.keys())
    if user_name in memory.keys():
        if input('Would you like to summarize your memory?If yes, please enter "yes"') == "yes":
            user_memory = summarize_memory_event_personality(data_args, memory, user_name)
    hello_msg,user_memory,memory,user_name,user_memory_index = enter_name(user_name,memory,local_memory_qa,data_args)
    print(hello_msg)
    # print('Would you like to summarize your memory?If yes, please enter "yes"')
    
    print("欢迎使用 SiliconFriend模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input(f"\n{user_name}：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 SiliconFriend模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        history_state, history, msg = predict_new(text=query,history=history,top_p=0.95,temperature=1,max_length_tokens=1024,max_context_length_tokens=200,user_name=user_name,user_memory=user_memory,user_memory_index=user_memory_index)
        if stop_stream:
                stop_stream = False
                break
        else:
            count += 1
            if count % 8 == 0:
                os.system(clear_command)
                print(output_prompt(history_state,user_name,boot_actual_name), flush=True)
                signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)       
        print(output_prompt(history_state,user_name,boot_actual_name), flush=True)

if __name__ == "__main__":
    main()