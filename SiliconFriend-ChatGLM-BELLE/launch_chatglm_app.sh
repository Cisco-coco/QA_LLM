export OPENAI_API_KEY=api-key
base_model=/data/sshfs/dataroot/models/THUDM/chatglm-6b
adapter_model=/data/sshfs/94code/MemoryBank-SiliconFriend/model/ChatGLM-LoRA-checkpoint
python app_demo.py \
    --base_model $base_model \
    --adapter_model $adapter_model \
    --language cn

