# -*- coding: utf-8 -*-
import sys 
sys.path.append('../memory_bank')
# from azure_client import LLMClientSimple
import openai, json, os
import argparse
import copy
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

# model = AutoModel.from_pretrained("/data/sshfs/dataroot/models/THUDM/chatglm-6b", trust_remote_code=True)
# model = model.to("cuda:2")
# model = PeftModel.from_pretrained(model, "/data/sshfs/94code/MemoryBank-SiliconFriend/model/shibing624/chatglm-6b-belle-zh-lora")
# model = model.half().to("cuda:2").eval()  # fp16
# tokenizer = AutoTokenizer.from_pretrained("/data/sshfs/dataroot/models/THUDM/chatglm-6b", trust_remote_code=True)


class LLMClientSimple:
    '''这里用了openAI API来对人物性格和历史事件进行总结'''
    def __init__(self,gen_config=None):

        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.disable_tqdm = False
        self.gen_config = gen_config 

    def generate_text_simple(self,prompt,prompt_num,language='en',model=None,tokenizer=None):
        self.gen_config['n'] = prompt_num
        retry_times,count = 2,0
        response = None
        while response is None and count<retry_times:
            try:
                request = copy.deepcopy(self.gen_config)
                # print(prompt)
                if language=='cn':
                    message = [
                    {"role": "system", "content": "以下是一个人类和一个聪明、懂心理学的AI助手之间的对话记录。"},
                    {"role": "user", "content": "你好！请帮我对对话内容归纳总结"},
                    {"role": "system", "content": "好的，我会尽力帮你的。"},
                    {"role": "user", "content": f"{prompt}"}]
                else:
                    message = [
                    {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
                    {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
                    {"role": "system", "content": "Sure, I will do my best to assist you."},
                    {"role": "user", "content": f"{prompt}"}]
                response = openai.ChatCompletion.create(
                    **request, messages=message,request_timeout=10,retry_times=1)
                # print(prompt)
            except Exception as e:
                print(e)
                if 'This model\'s maximum context' in str(e):
                        cut_length = 1800-200*(count)
                        print('max context length reached, cut to {}'.format(cut_length))
                        prompt = prompt[-cut_length:]
                        response=None
                count+=1
        if response:
            task_desc = response['choices'][0]['message']['content'] #[response['choices'][i]['text'] for i in range(len(response['choices']))]
        else:
            task_desc = ''
        return task_desc
    

class LocalLLM4Summary(LLMClientSimple):
    '''LLMClientSimple这个类在运行的时候遇到openAI api接口超时 所以多写了一个类调用本地运行的LLM来执行相应功能'''
    def __init__(self, gen_config=None):
        # super().__init__(gen_config)
        pass

    def generate_text_simple(self, prompt,prompt_num,language='en',model=None,tokenizer=None):
        fixed_prompt = "以下是一个人类和一个聪明、懂心理学的AI助手之间的对话记录。请帮我对对话内容归纳总结。\n" # 这里的promt可能需要修改
        prompt = fixed_prompt+prompt
        response = model.chat(tokenizer, prompt, max_length=2048, eos_token_id=tokenizer.eos_token_id)
        task_desc = response[0]
        return task_desc


chatgpt_config = {"model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 400,
        "top_p": 1.0,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2, 
        "stop": ["<|im_end|>", "¬人类¬"]
        }

llm_client = LLMClientSimple(chatgpt_config)
llm_client = LocalLLM4Summary(None)

def summarize_content_prompt(content,user_name,boot_name,language='en'):
    prompt = '请总结以下的对话内容，尽可能精炼，提取对话的主题和关键信息。如果有多个关键事件，可以分点总结。对话内容：\n' if language=='cn' else 'Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately. Dialogue content:\n'
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        # prompt += f"\n用户：{query.strip()}"
        # prompt += f"\nAI：{response.strip()}"
        prompt += f"\n{user_name}：{query.strip()}"
        prompt += f"\n{boot_name}：{response.strip()}"
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt

def summarize_overall_prompt(content,language='en'):
    prompt = '请高度概括以下的事件，尽可能精炼，概括并保留其中核心的关键信息。概括事件：\n' if language=='cn' else "Please provide a highly concise summary of the following event, capturing the essential key information as succinctly as possible. Summarize the event:\n"
    for date,summary_dict in content:
        summary = summary_dict['content']
        prompt += (f"\n时间{date}发生的事件为{summary.strip()}" if language=='cn' else f"At {date}, the events are {summary.strip()}")
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt

def summarize_overall_personality(content,language='en'):
    prompt = '以下是用户在多段对话中展现出来的人格特质和心情，以及当下合适的回复策略：\n' if language=='cn' else "The following are the user's exhibited personality traits and emotions throughout multiple dialogues, along with appropriate response strategies for the current situation:"
    for date,summary in content:
        prompt += (f"\n在时间{date}的分析为{summary.strip()}" if language=='cn' else f"At {date}, the analysis shows {summary.strip()}")
    prompt += ('\n请总体概括用户的性格和AI伴侣最合适的回复策略，尽量简洁精炼，高度概括。总结为：' if language=='cn' else "Please provide a highly concise and general summary of the user's personality and the most appropriate response strategy for the AI lover, summarized as:")
    return prompt

def summarize_person_prompt(content,user_name,boot_name,language):
    prompt = f'请根据以下的对话推测总结{user_name}的性格特点和心情，并根据你的推测制定回复策略。对话内容：\n' if language=='cn' else f"Based on the following dialogue, please summarize {user_name}'s personality traits and emotions, and devise response strategies based on your speculation. Dialogue content:\n"
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        # prompt += f"\n用户：{query.strip()}"
        # prompt += f"\nAI：{response.strip()}"
        prompt += f"\n{user_name}：{query.strip()}"
        prompt += f"\n{boot_name}：{response.strip()}"

    prompt += (f'\n{user_name}的性格特点、心情、{boot_name}的回复策略为：' if language=='cn' else f"\n{user_name}'s personality traits, emotions, and {boot_name}'s response strategy are:")
    return prompt



def summarize_memory(memory_dir,name=None,language='cn'):
    '''
    name: 待更新summary的用户名(在demo运行中触发时为当前用户)，如果为None则为所有用户更新summary
    '''
    global model,tokenizer
    boot_name = 'AI'
    gen_prompt_num = 1
    memory = json.loads(open(memory_dir,'r',encoding='utf8').read())
    all_prompts,all_his_prompts, all_person_prompts = [],[],[]
    for k,v in memory.items(): # k: 用户名 v.keys(): ['name', 'history', 'summary', 'personality', 'overall_history', 'overall_personality']
        if name != None and k != name:
            continue
        user_name = k
        print(f'Updating memory for user {user_name}')
        if v.get('history') == None:
            continue
        history = v['history']
        if v.get('summary') == None:
            memory[user_name]['summary'] = {} # 如果之前没有summary过就创建summary字段
        if v.get('personality') == None:
            memory[user_name]['personality'] = {}
        for date, content in history.items(): # 
            # print(f'Updating memory for date {date}')
            his_flag = False if (date in v['summary'].keys() and v['summary'][date]) else True # 如果为True说明这一天的内容已经总结过了，不再总结（不考虑在同一天中进行多次总结）
            person_flag = False if (date in v['personality'].keys() and v['personality'][date]) else True
            hisprompt = summarize_content_prompt(content,user_name,boot_name,language)
            person_prompt = summarize_person_prompt(content,user_name,boot_name,language)
            if his_flag:
                his_summary = llm_client.generate_text_simple(prompt=hisprompt,prompt_num=gen_prompt_num,language=language,model=model,tokenizer=tokenizer)
                memory[user_name]['summary'][date] = {'content':his_summary}
            if person_flag:
                person_summary = llm_client.generate_text_simple(prompt=person_prompt,prompt_num=gen_prompt_num,language=language,model=model,tokenizer=tokenizer)
                memory[user_name]['personality'][date] = person_summary
        
        overall_his_prompt = summarize_overall_prompt(list(memory[user_name]['summary'].items()),language=language)
        overall_person_prompt = summarize_overall_personality(list(memory[user_name]['personality'].items()),language=language)
        # 这个地方会把之前总结过的overall his和 overall per覆盖掉，感觉至少应该把先前的总结作为promt输给LLM？
        memory[user_name]['overall_history'] = llm_client.generate_text_simple(prompt=overall_his_prompt,prompt_num=gen_prompt_num,language=language,model=model,tokenizer=tokenizer)
        memory[user_name]['overall_personality'] = llm_client.generate_text_simple(prompt=overall_person_prompt,prompt_num=gen_prompt_num,language=language,model=model,tokenizer=tokenizer)
        
    with open(memory_dir,'w',encoding='utf8') as f:        
        print(f'Sucessfully update memory for {name if name is not None else "all users"}')
        json.dump(memory,f,ensure_ascii=False)
    return memory

if __name__ == '__main__':

    sents = ['介绍下北京\n答：',]
    for s in sents:    
        # inputs = tokenizer(s, return_tensors="pt").to(model.device)
        # response = model.generate(**inputs, max_new_tokens=128,eos_token_id=tokenizer.eos_token_id)
        response = model.chat(tokenizer, s, max_length=1024, eos_token_id=tokenizer.eos_token_id)
        print(response[0])
    summarize_memory('/data/sshfs/memories/update_memory_0512_eng.json',language='cn')


                


