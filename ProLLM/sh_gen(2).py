#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime

# 基本配置
BASE_PATH = "/home/wuwujian/LXY/sensor_process/seedLLM/benchmark/ProLLM"
EXPERIMENT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 需要并行运行的不同随机种子
SEED_LIST = [3407, 42069, 24601]  # ✅ 你可以自由修改，比如 [1, 11, 21, 31]

def generate_sh_scripts():
    """生成实验脚本文件（支持多seed版本）"""
    print("\n=== 开始生成实验脚本（多seed版本） ===")
    
    # 脚本模板
    script_template = """export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
python -u main.py \\
  --epochs 100 \\
  --batch_size 32 \\
  --seed {seed} \\
  --lr 1e-5 \\
  --gamma 0.8 \\
  --step_size 50 \\
  --dataset {dataset_name} \\
  --length 361 \\
  --dimensions 6 \\
  --num_class 4 \\
  --few_shot 0 \\
  --llm_type gpt2 \\
  --lora 1 \\
  --patch_len 32 \\
  --stride 16 \\
  --channels 256 \\
  --depth 3 \\
  --reduced_size 128 \\
  --kernel_size 2 \\
  --path SS_ckpt \\
  --wandb_project LLM \\
  --wandb_entity 1343921617-0 \\
  --wandb_run_name TimeDG_con{source_con}_con{target_con}_seed{seed} \\
  --use_channel_attention \\
  --channel_attention_type multiscale \\
  --attention_position before \\
    --use_offline_embeddings \
    --generate_embeddings \
    --prompt_representation pooled_last_token \
  --llm_path gpt2 \\
  --tokenizer_path gpt2
"""
    
    scripts_info = []

    # 循环生成不同源域-目标域组合及不同seed的脚本
    for source_con in range(1, 7):  # 源域浓度 (1-6)
        for target_con in range(1, 7):  # 目标域浓度 (1-6)
            for seed in SEED_LIST:
                experiment_name = f"con{source_con}con{target_con}_seed{seed}"
                
                # 替换模板参数
                script_content = script_template.format(
                    seed=seed,
                    dataset_name=f"con{source_con}con{target_con}Sensor",
                    source_con=source_con,
                    target_con=target_con
                )
                
                # 文件名包含seed
                script_filename = f"Sensor_con{source_con}_con{target_con}_seed{seed}.sh"
                script_path = os.path.join(BASE_PATH, script_filename)
                
                # 写入文件
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # 设为可执行
                os.chmod(script_path, 0o755)
                
                scripts_info.append({
                    "script_path": script_path,
                    "script_filename": script_filename,
                    "source_con": source_con,
                    "target_con": target_con,
                    "seed": seed
                })
                
                print(f"✅ 已生成脚本: {script_filename}")
    
    # 生成 Sensor 数据集的脚本
    for seed in SEED_LIST:
        experiment_name = f"Sensor_seed{seed}"
        
        # 替换模板参数
        script_content = script_template.format(
            seed=seed,
            dataset_name="Sensor",
            source_con="Sensor",  # 使用 Sensor 作为标识
            target_con="Sensor"
        )
        
        # 替换 wandb_run_name 为 Sensor 版本
        script_content = script_content.replace(
            "--wandb_run_name TimeDG_conSensor_conSensor_seed{seed}",
            f"--wandb_run_name TimeDG_Sensor_seed{seed}"
        )
        
        # 文件名包含seed
        script_filename = f"Sensor_seed{seed}.sh"
        script_path = os.path.join(BASE_PATH, script_filename)
        
        # 写入文件
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 设为可执行
        os.chmod(script_path, 0o755)
        
        scripts_info.append({
            "script_path": script_path,
            "script_filename": script_filename,
            "source_con": "Sensor",
            "target_con": "Sensor",
            "seed": seed
        })
        
        print(f"✅ 已生成脚本: {script_filename}")
    
    return scripts_info

def generate_run_all_script(scripts_info):
    """生成总执行脚本（支持多seed版本）"""
    run_all_content = f"""#!/bin/bash

# 自动化运行所有浓度泛化实验（包含不同seed）
# 生成时间: {EXPERIMENT_TIMESTAMP}

"""
    
    for info in scripts_info:
        sc = info["source_con"]
        tc = info["target_con"]
        seed = info["seed"]
        script_filename = info["script_filename"]
        
        run_all_content += f"""
echo "===== 开始运行: 源域={sc}, 目标域={tc}, seed={seed} ====="
bash {script_filename}
echo "===== 完成运行: 源域={sc}, 目标域={tc}, seed={seed} ====="
"""
    
    run_all_path = os.path.join(BASE_PATH, f"run_all_experiments_{EXPERIMENT_TIMESTAMP}.sh")
    with open(run_all_path, 'w') as f:
        f.write(run_all_content)
    
    # 设置权限
    os.chmod(run_all_path, 0o755)
    
    print(f"\n🎯 已生成总运行脚本: run_all_experiments_{EXPERIMENT_TIMESTAMP}.sh")
    return run_all_path

def main():
    """主函数"""
    print("开始生成所有多seed实验脚本...")
    scripts_info = generate_sh_scripts()
    run_all_path = generate_run_all_script(scripts_info)
    
    print("\n=== 所有脚本生成完成! ===")
    print(f"运行以下命令可开启所有实验:")
    print(f"  bash {os.path.basename(run_all_path)}")
    print("\n或单独运行，如:")
    print("  bash Sensor_con1_con2_seed29.sh")

if __name__ == "__main__":
    main()

