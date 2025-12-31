import json
import torch
from transformers import AutoConfig, AutoModel
from peft import LoraConfig, get_peft_model

def LLMprepare(configs):
    # Load model configurations from JSON file
    try:
        with open('llm_config.json', 'r') as f:
            model_configurations = json.load(f)
        
        config = model_configurations.get(configs.llm_type)
        if not config:
            raise ValueError("Unsupported LLM type")
        
        model_path = config["path"]
        d_model = config["dim"]
        model_class = eval(config["model_class"])
        
        llm_model = model_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16
        )
        
        # Apply LoRA modifications if requested
        if configs.lora:
            lora_config = config['lora_config']
            lora_config['lora_dropout'] = configs.dropout  # Update dropout with user input
            llm_model = get_peft_model(llm_model, LoraConfig(**lora_config))
            for name, param in llm_model.named_parameters():
                param.requires_grad = ('lora' in name)
        else:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
    
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        print("Warning: llm_config.json not found or invalid. Using GPT2 as fallback.")
        # Fallback to GPT2 if configuration file is not available
        llm_model = AutoModel.from_pretrained('gpt2')
        d_model = llm_model.config.n_embd
        
        # Freeze parameters
        for name, param in llm_model.named_parameters():
            param.requires_grad = False
    
    return llm_model, d_model