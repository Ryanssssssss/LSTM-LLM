#!/bin/bash

# 预处理所有数据集的embeddings
# 此脚本会遍历所有需要的数据集配置，生成并保存prompt embeddings

# 设置设备
DEVICE="cuda"
TOKENIZER_PATH="gpt2"
LLM_PATH="gpt2"

# 所有浓度和折叠的组合
for SOURCE_CON in 1 2 3 4 5 6; do
    for TARGET_CON in 1 2 3 4 5 6; do
        # 跳过相同浓度
        if [ $SOURCE_CON -eq $TARGET_CON ]; then
            continue
        fi
        
        for FOLD in 1 2 3 4 5; do
            DATASET="con${SOURCE_CON}con${TARGET_CON}_fold${FOLD}"
            echo "===== 处理数据集: ${DATASET} ====="
            
            # 检查数据集目录是否存在
            if [ -d "npydata/${DATASET}" ]; then
                python embed_prompt.py \
                    --dataset ${DATASET} \
                    --device ${DEVICE} \
                    --tokenizer_path ${TOKENIZER_PATH} \
                    --llm_path ${LLM_PATH} \
                    --representation pooled_last_token
                
                echo "===== 完成数据集: ${DATASET} 的embedding生成 ====="
            else
                echo "!!! 警告: 数据集 ${DATASET} 目录不存在，跳过 !!!"
            fi
        done
    done
done

echo "所有数据集的embeddings生成完成!"