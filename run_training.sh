#!/bin/bash
# 自动运行训练并将输出保存到logs目录

# 创建logs目录
mkdir -p logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "开始训练..."
echo "日志将保存到: ${LOG_FILE}"

# 运行训练并保存日志
python3 LSTM-LLM.py 2>&1 | tee "${LOG_FILE}"

echo ""
echo "训练完成！"
echo "训练日志: ${LOG_FILE}"
echo "JSON日志: logs/lstm_llm_${TIMESTAMP}.json"
