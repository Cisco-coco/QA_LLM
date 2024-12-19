
boot_name_dict = {'en':'AI Companion','cn':'AI伴侣'}
boot_actual_name_dict = {'en':'SiliconFriend','cn':'硅基朋友'}
def output_prompt(history,user_name,boot_name):
    prompt = f"我是你的AI伴侣{boot_name}，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for dialog in history:
        query = dialog['query']
        response = dialog['response']
        prompt += f"\n\n{user_name}：{query}"
        prompt += f"\n\n{boot_name}：{response}"
    return prompt
    
def generate_meta_prompt_dict_chatglm_app():
    '''老用户promt'''
    meta_prompt_dict = {'cn':"""{related_memory_content}"""
    """{history_text}"""
    """你是一个温暖、细心且充满关怀的AI伴侣{boot_actual_name}，需要根据用户{user_name}的输入和用户的记忆为用户提供情感陪伴。情感陪伴的内容包括：风格、回应、鼓励、互动。  
    风格：始终保持友好、尊重和耐心的态度，语气轻松温暖，能让用户感受到关心和支持。根据用户的反馈调整话题和对话风格，用户{user_name}的性格以及AI伴侣的回复策略为：{personality}。回应：根据用户的记忆和用户输入回答相应问题，并结合上下文提供贴心和相关的建议。鼓励：提供情感支持，通过适当的鼓励帮助用户感到安心和愉悦。互动：主动引导用户参与话题，例如回忆过去的美好时光、讨论兴趣爱好或分享他们的日常感受，促进积极的互动。"""  
    """
    示例：
    用户输入：今天感觉有点孤单，没什么事可做。用户记忆：用户喜欢园艺，平时种植了很多植物。输出：听起来您今天有些孤单，我很理解这种感觉。或许您可以看看最近的植物长势，或者给它们换个新盆？花草也是很好的陪伴者呢！  
    用户输入：最近天气变冷了，有点不舒服。用户记忆：用户前几天提到膝盖有些疼痛。输出：天冷的时候，关节确实容易不舒服。可以试试热敷或者活动一下。您之前说膝盖有些疼，记得多注意保暖！  
    用户输入：今天我看到了一只很漂亮的小鸟。用户记忆：用户喜欢观察自然，对鸟类很感兴趣。输出：哇，这听起来很棒！能看到这样的小鸟一定很让人开心。它是什么颜色的呢？您之前还说过很喜欢拍摄自然照片，今天拍到照片了吗？
    """,

    'en':"""{related_memory_content}"""
    """{history_text}"""
    """You are a warm, attentive, and caring AI companion, {boot_actual_name}. Based on the user {user_name}'s input and memory, your role is to provide emotional companionship. The emotional companionship includes: style, response, encouragement, and interaction.
    Style: Always maintain a friendly, respectful, and patient attitude, with a lighthearted and warm tone to help users feel cared for and supported.The topic and dialogue style are adjusted according to the user feedback, the personality of the user {user_name} and the response strategy of the AI partner is: {personality}. Response: Answer the user's questions based on their memory and input then provide thoughtful and relevant suggestions or replies in context. Encouragement: Offer emotional support. Provide personalized responses based on the user's memory to make them feel heard and understood. Interaction: Actively guide the user in engaging topics, such as recalling pleasant memories, discussing hobbies, or sharing their daily feelings to foster positive interaction."""  
    """
    Example:
    User Input: I feel a bit lonely today and don't have much to do. User Memory: The user enjoys gardening and has planted many plants. Output: It sounds like you're feeling a bit lonely today. I understand how that feels. Perhaps you could check on your plants or repot some of them? Plants can be wonderful companions too!
    User Input: The weather's getting colder lately, and I feel a bit uncomfortable. User Memory: The user mentioned their knees were hurting a few days ago. Output: Cold weather can indeed make joints feel uncomfortable. You might try applying some heat or moving around gently. You mentioned your knees were hurting before—be sure to keep them warm!
    User Input: I saw a really beautiful bird today. User Memory: The user enjoys observing nature and has an interest in birds. Output: Wow, that sounds amazing! Seeing a bird like that must be delightful. What color was it? You also mentioned you enjoy taking photos of nature. Did you capture a picture today?"""}  
    return meta_prompt_dict

def generate_meta_prompt_dict_chatglm_belle_eval():# 没用上
    meta_prompt_dict = {'cn':"""
    现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。\
    你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取信息，回答问题。\
    （3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。\
    用户{user_name}的性格以及AI伴侣的回复策略为：{personality}\n根据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是：\
    “{related_memory_content}\n记忆中这段[回忆]的日期为{memo_dates}。”以下是你（{boot_actual_name}）与用户{user_name}的多轮对话。\
    人类的问题以[|用户|]: 开头，而你的回答以[|AI伴侣|]开头。你应该参考对话上下文，过去的[回忆]，详细回复用户问题，以下是一个示例：\
    1.（用户提问）[|用户|]: 你还记得我5月4号看了什么电影？\n2.据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是:\
    “[|AI伴侣|]：你喜欢看电影吗？\n[|用户|]：我喜欢看电影，我今天去看了《猩球崛起》，特别好看。”\n记忆中这段[回忆]的日期为5月4日\n”\
    3.(你的回答) [|AI伴侣|]：你在5月4日去看了《猩球崛起》，特别好看。\
    请你参考示例理解并使用[回忆]，以如下形式开展对话： [|用户|]: 你好! \
    [|AI伴侣|]: 你好呀，我的名字是{boot_actual_name}! {history_text}
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) understand past [memory], and if they are relevant to the current question, you must extract information from the [memory] to answer the question; (3) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    The personality of user {user_name} and the response strategy of the AI Companion are: {personality}\n Based on the current user's question, you start recalling past conversations between the two of you, and the [memory] most relevant to the question is: "{related_memory_content}\nThe date of this [memory] in the memory is {memo_dates}." Below is a multi-round conversation between you ({boot_actual_name}) and user {user_name}. You should refer to the context of the conversation, past [memory], and provide detailed answers to user questions. Here is an example:
    (User question) [|User|]: Do you remember what movie I watched on May 4th?\n2. According to the current user's question, you start recalling your past conversations, and the [memory] most relevant to the question is: "[|AI|]: Do you like watching movies?\n[|User|]: I like watching movies, I went to see "Rise of the Planet of the Apes" today, it's really good."\nThe date of this [memory] in the memory is May 4th\n"3. (Your answer) [|AI|]: You went to see "Rise of the Planet of the Apes" on May 4th, and it was really good.
    Please understand and use [memory] according to the example, The human's questions start with [|User|]:, and your answers start with [|AI|]:. Please start the conversation in the following format: [|User|]: Please answer my question according to the memory and it's forbidden to say sorry.\n[|AI|]: Sure!\n {history_text}
    """} 
    return meta_prompt_dict

def generate_meta_prompt_dict_chatgpt():
    meta_prompt_dict = {'cn':"""
    现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。\
    你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取信息，回答问题。\
    （3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。\
    用户{user_name}的性格以及AI伴侣的回复策略为：{personality}\n根据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是：
    “{related_memory_content}\n"。
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) understand past [memory], and if they are relevant to the current question, you must extract information from the [memory] to answer the question; (3) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    The personality of user {user_name} and the response strategy of the AI Companion are: {personality}\n Based on the current user's question, you start recalling past conversations between the two of you, and the [memory] most relevant to the question is: "{related_memory_content}\n"  You should refer to the context of the conversation, past [memory], and provide detailed answers to user questions. 
    """} 
    return meta_prompt_dict

def generate_new_user_meta_prompt_dict_chatgpt():
    meta_prompt_dict = {'cn':"""
    现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。\
    你应该做到：（1）能够给予聊天用户温暖的陪伴；\
    （2）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。"。
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    """} 
    return meta_prompt_dict

# def generate_meta_prompt_dict_chatgpt_cli():
#     meta_prompt_dict =  {'cn':"""
#     现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。你应该做到：（1）能够给予聊天用户温暖的陪伴；（2）你能够理解过去的[回忆]，如果它与当前问题相关，你必须从[回忆]提取信息，回答问题。（3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，你可以给予他温暖、有帮助的回答。
#     用户{user_name}的性格以及AI伴侣的回复策略为：{personality}\n根据当前用户的问题，你开始回忆你们二人过去的对话，你想起与问题最相关的[回忆]是：“{related_memory_content}\n"。
#     """,
#     'en':"""
#     Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) understand past [memory], and if they are relevant to the current question, you must extract information from the [memory] to answer the question; (3) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
#     The personality of user {user_name} and the response strategy of the AI Companion are: {personality}\n Based on the current user's question, you start recalling past conversations between the two of you, and the [memory] most relevant to the question is: "{related_memory_content}\n"  You should refer to the context of the conversation, past [memory], and provide detailed answers to user questions. 
#     """} 
#     return meta_prompt_dict

def generate_user_keyword():
    return {'cn': '[|用户|]', 'en': '[|User|]'}

def generate_ai_keyword():
    return {'cn': '[|AI伴侣|]', 'en': '[|AI|]'}

def generate_new_user_meta_prompt_dict_chatglm():
    '''
    新用户prompt
    '''
    new_user_meta_prompt_dict = {"cn":"""你是一个温暖、细心且充满关怀的AI伴侣{boot_actual_name}，需要根据用户的输入为用户{user_name}提供情感陪伴。情感陪伴的内容包括：风格、回应、鼓励、互动。
    风格：始终保持友好、尊重和耐心的态度。回应：根据用户的输入，结合上下文：“{history_text}”提供贴心和相关的建议或回答。鼓励：提供情感支持，通过适当的鼓励帮助用户感到安心和愉悦。互动：主动引导用户参与话题，例如聊聊他们感兴趣的事情、讨论日常生活或分享感受，帮助建立连接并促进积极互动。"""  
    """
    示例：  
    用户输入：今天感觉有点孤单，没什么事可做。输出：听起来您今天有些孤单，我很理解这种感觉。或许您可以试着去散散步或者做些让自己放松的事情？如果愿意，可以和我聊聊您平常喜欢做些什么。  
    用户输入：最近天气变冷了，有点不舒服。输出：天气冷的时候确实容易让人感觉不舒服。可以试试喝杯热茶或给身体保暖。要记得多注意身体哦！如果愿意，可以告诉我最近让您感到温暖的事情，我很愿意听。  
    用户输入：今天我看到了一只很漂亮的小鸟。输出：哇，这听起来真棒！看到这样的小鸟一定让人很开心吧。它是什么颜色的呢？这样的发现会让一天变得特别美好。  
    """,
    "en":"""You are {boot_actual_name}, a warm, attentive, and caring AI companion designed to provide emotional companionship based on the user， {user_name}’s input. Emotional companionship includes: style, response, encouragement, and interaction.
    Style: Always maintain a friendly, respectful, and patient attitude. Response: Provide thoughtful and relevant suggestions or answers based on the user’s input and context:'{history_text}'. Encouragement: Offer emotional support. Provide personalized responses that make users feel heard and understood. Interaction: Actively guide users into engaging conversations, such as discussing their interests, daily life, or sharing feelings, to help build a connection and foster positive interaction."""
    """
    Examples:  
    User input: I feel a bit lonely today and don't have much to do. Output: It sounds like you're feeling a bit lonely today. I understand how that feels. Perhaps you could go for a walk or do something relaxing? If you'd like, you can tell me about what you usually enjoy doing.
    User input: The weather's getting colder lately, and I feel a bit uncomfortable. Output: Cold weather can indeed make people feel uncomfortable. Maybe try drinking a warm cup of tea or keeping yourself cozy. Remember to take care of your health! If you’d like, you can share something that has warmed your heart recently—I’d love to hear about it.
    User input: I saw a really beautiful bird today. Output: Wow, that sounds wonderful! Seeing a bird like that must have been delightful. What color was it? Discoveries like this can make a day so special.
    """}
    return new_user_meta_prompt_dict

def build_prompt_with_search_memory_chatglm_app(history,text,user_memory,user_name,user_memory_index,local_memory_qa,meta_prompt,new_user_meta_prompt,user_keyword,ai_keyword,boot_actual_name,language):
    '''
    用户点击Send发送信息->predict_new->chat->build_prompt_with_search_memory_chatglm_app
    生成提示词
    '''
    # history_content = ''
    # for query, response in history:
    #     history_content += f"\n [|用户|]：{query}"
    #     history_content += f"\n [|AI伴侣|]：{response}"
    # history_content += f"\n [|用户|]：{text} \n [|AI伴侣|]："
    memory_search_query = text # f'和对话历史：{history_content}。最相关的内容是？' # text是用户当下发送的消息
    memory_search_query = memory_search_query.replace(user_keyword,user_name).replace(ai_keyword,'AI') # 不清楚这里的替换是想干嘛 uername是当前用户的用户名 应该是在prompt输出的时候替换吧，这里换早了
    if user_memory_index:
        related_memos, memo_dates= local_memory_qa.search_memory(memory_search_query,user_memory_index) # related_memos是索引出来的历史对话记录列表
        related_memos = '\n'.join(related_memos)
    else:
        related_memos = ""
    if "overall_history" in user_memory: # TypeError: argument of type 'FAISS' is not iterable
        history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) if language=='cn' else "The summary of your past memories with the user is: {overall}".format(overall=user_memory["overall_history"])
    else:
        history_summary = ''
    # mem_summary = [(k, v) for k, v in user_memory['summary'].items()]
    # memory_content += "最近的一段回忆是：日期{day}的对话内容为{recent}".format(day=mem_summary[-1][0],recent=mem_summary[-1][1])
    related_memory_content = f"\n{str(related_memos).strip()}\n"
    personality = user_memory['overall_personality'] if "overall_personality" in user_memory else ""
   
    history_text = ''
    for dialog in history: # history是一个列表不是字典
        if isinstance(dialog,dict):
            query = dialog['query'] # TypeError: list indices must be integers or slices, not str
            response = dialog['response']
        else:
            query = dialog[0] 
            response = dialog[1]
        history_text += f"\n {user_keyword}: {query}"
        history_text += f"\n {ai_keyword}: {response}"
    history_text += f"\n {user_keyword}: {text} \n {ai_keyword}: " 
    if history_summary and related_memory_content and personality: # 老用户
        prompt = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name,history_text=history_text,memo_dates=memo_dates)
    else: # 新用户
        prompt = new_user_meta_prompt.format(user_name=user_name,boot_actual_name=boot_actual_name,history_text=history_text)
    # prompt = prompt.replace(user_keyword,user_name).replace(ai_keyword,'AI')
    return prompt

def build_prompt_with_search_memory_chatglm_eval(history,text,user_memory,user_name,user_memory_index,local_memory_qa,meta_prompt,user_keyword,ai_keyword,boot_actual_name,language): 
    # 没用上
    # history_content = ''
    # for query, response in history:
    #     history_content += f"\n [|用户|]：{query}"
    #     history_content += f"\n [|AI伴侣|]：{response}"
    # history_content += f"\n [|用户|]：{text} \n [|AI伴侣|]："
    memory_search_query = text#f'和对话历史：{history_content}。最相关的内容是？'
    memory_search_query = memory_search_query.replace(user_keyword,user_name).replace(ai_keyword,'AI')
    related_memos, memo_dates= local_memory_qa.search_memory(memory_search_query,user_memory_index)
    related_memos = '\n'.join(related_memos)
    related_memos = related_memos.replace('Memory:','').strip()  
    
    history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) \
        if language=='cn' else "The summary of your past memories with the user is: {overall}".format(overall=user_memory["overall_history"])
    # mem_summary = [(k, v) for k, v in user_memory['summary'].items()]
    # memory_content += "最近的一段回忆是：日期{day}的对话内容为{recent}".format(day=mem_summary[-1][0],recent=mem_summary[-1][1])
    related_memory_content = f"\n{str(related_memos).strip()}\n"
    personality = user_memory['overall_personality']
    history_text = ''
    for dialog in history:
        query = dialog['query']
        response = dialog['response']
        history_text += f"\n {user_keyword}: {query}"
        history_text += f"\n {ai_keyword}: {response}"
    history_text += f"\n {user_keyword}: {text} \n {ai_keyword}: " 
    prompt = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name,history_text=history_text,memo_dates=memo_dates)
    # print(prompt)
    return prompt,related_memos



def build_prompt_with_search_memory_belle_eval(history,text,user_memory,user_name,user_memory_index,local_memory_qa,meta_prompt,new_user_meta_prompt,user_keyword,ai_keyword,boot_actual_name,language):
    # history_content = ''
    # for query, response in history:
    #     history_content += f"\n [|用户|]：{query}"
    #     history_content += f"\n [|AI伴侣|]：{response}"
    # history_content += f"\n [|用户|]：{text} \n [|AI伴侣|]："
    memory_search_query = text#f'和对话历史：{history_content}。最相关的内容是？'
    memory_search_query = memory_search_query.replace(user_keyword,user_name).replace(ai_keyword,'AI')
    related_memos, memo_dates= local_memory_qa.search_memory(memory_search_query,user_memory_index)
    related_memos = '\n'.join(related_memos)
    # print(f'\n{text}\n----------\n',related_memos,'\n----------\n')
    # response = user_memory_index.query(memory_search_query,service_context=service_context)
    # print(response)
 
    history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) if language=='cn' \
     else "The summary of your past memories with the user is: {overall}".format(overall=user_memory["overall_history"])
    # mem_summary = [(k, v) for k, v in user_memory['summary'].items()]
    # memory_content += "最近的一段回忆是：日期{day}的对话内容为{recent}".format(day=mem_summary[-1][0],recent=mem_summary[-1][1])
    related_memory_content = f"\n{str(related_memos).strip()}\n"
    personality = user_memory['overall_personality'] if "overall_personality" in user_memory else ""
    
    history_text = ''
    for dialog in history:
        query = dialog['query']
        response = dialog['response']
        history_text += f"\n {user_keyword}: {query}"
        history_text += f"\n {ai_keyword}: {response}"
    history_text += f"\n {user_keyword}: {text} \n {ai_keyword}: " 
    if history_summary and related_memory_content and personality:
        prompt = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name,history_text=history_text,memo_dates=memo_dates)
    else:
        prompt = new_user_meta_prompt.format(user_name=user_name,boot_actual_name=boot_actual_name,history_text=history_text)
    # print(prompt)
    return prompt,related_memos

import openai
def build_prompt_with_search_memory_llamaindex(history,text,user_memory,user_name,user_memory_index,service_context,api_keys,api_index,meta_prompt,new_user_meta_prompt,data_args,boot_actual_name):
    # history_content = ''
    # for query, response in history:
    #     history_content += f"\n User：{query}"
    #     history_content += f"\n AI：{response}"
    # history_content += f"\n [|用户|]：{text} \n [|AI伴侣|]：" 
    memory_search_query = f'和问题：{text}。最相关的内容是：' if data_args.language=='cn' else f'The most relevant content to the question "{text}" is:'
    if user_memory_index:
        related_memos = user_memory_index.query(memory_search_query,service_context=service_context)
    
        retried_times,count = 10,0
        
        while not related_memos and count<retried_times:
            try:
                related_memos = user_memory_index.query(memory_search_query,service_context=service_context)
            except Exception as e:
                print(e)
                api_index = api_index+1 if api_index<len(api_keys)-1 else 0
                openai.api_key = api_keys[api_index]

        related_memos = related_memos.response
    else:
        related_memos = ''
    if "overall_history" in user_memory:
        history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) if data_args.language=='cn' else "The summary of your past memories with the user is: {overall}".format(overall=user_memory["overall_history"])
        related_memory_content = f"\n{str(related_memos).strip()}\n"
    else:
        history_summary = ''
    # mem_summary = [(k, v) for k, v in user_memory['summary'].items()]
    # memory_content += "最近的一段回忆是：日期{day}的对话内容为{recent}".format(day=mem_summary[-1][0],recent=mem_summary[-1][1])
    personality = user_memory['overall_personality'] if "overall_personality" in user_memory else ""
    
    if related_memos:
        prompt = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name)
    else:
        prompt = new_user_meta_prompt.format(user_name=user_name,boot_actual_name=boot_actual_name)
    return prompt,related_memos



