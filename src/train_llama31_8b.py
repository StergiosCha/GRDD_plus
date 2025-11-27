import os
from huggingface_hub import login

# Login with token first
HF_TOKEN = ''
os.environ['HF_TOKEN'] = HF_TOKEN

try:
    login(token=HF_TOKEN)
    print("✅ Successfully logged in to HuggingFace")
except Exception as e:
    print(f"Login warning: {e}")

# Set persistent storage paths
PERSISTENT_MODEL_DIR = os.path.expanduser("~/persistent_models")
os.environ['HF_HOME'] = f"{PERSISTENT_MODEL_DIR}/hf_cache"
os.environ['TRANSFORMERS_CACHE'] = f"{PERSISTENT_MODEL_DIR}/hf_cache"

import json
import torch
from transformers import (
    LlamaForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

def load_natural_prompts_dataset():
    texts = []
    
    with open('natural_prompts_dialects_only.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'prompt' in data and 'completion' in data:
                    text = f"{data['prompt']}\n{data['completion']}"
                    texts.append(text.strip())
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(texts)} natural prompt examples")
    return texts

def main():
    print("LLAMA-3.1-8B-INSTRUCT + LORA TRAINING")
    print("=" * 60)
    
    # Ensure persistent directories exist
    os.makedirs(f"{PERSISTENT_MODEL_DIR}/hf_cache", exist_ok=True)
    os.makedirs(f"{PERSISTENT_MODEL_DIR}/models", exist_ok=True)
    
    # Load dataset
    texts = load_natural_prompts_dataset()
    dataset = Dataset.from_dict({'text': texts})
    split_dataset = dataset.train_test_split(test_size=0.05)
    
    print("Loading Llama-3.1-8B-Instruct...")
    base_model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=HF_TOKEN,
        low_cpu_mem_usage=True
    )
    print("✅ Llama-3.1-8B model loaded successfully")
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        inference_mode=False
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", 
        use_auth_token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        result = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
        result['labels'] = result['input_ids'].copy()
        return result
    
    train_dataset = split_dataset['train'].map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = split_dataset['test'].map(tokenize_function, batched=True, remove_columns=['text'])
    
    # PERSISTENT STORAGE: Save to EFS
    output_dir = f"{PERSISTENT_MODEL_DIR}/models/llama31-dialects-lora"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        bf16=True,
        logging_steps=25,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        remove_unused_columns=False,
        report_to=[],
        warmup_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    print("Starting LoRA training on Llama-3.1-8B...")
    trainer.train()
    
    # Save to persistent storage
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model saved to: {output_dir}")
    print("Training complete!")

if __name__ == "__main__":
    main()
