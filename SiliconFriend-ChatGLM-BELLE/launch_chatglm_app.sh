export OPENAI_API_KEY=sk-vLIYD4EXBLP06GgvOq7Mh1mf2VyAxggRSBq7BnXw8kxQ06j6
base_model=/data/sshfs/dataroot/models/THUDM/chatglm-6b
adapter_model=/data/sshfs/94code/MemoryBank-SiliconFriend/model/ChatGLM-LoRA-checkpoint
python app_demo.py \
    --base_model $base_model \
    --adapter_model $adapter_model \
    --language cn

