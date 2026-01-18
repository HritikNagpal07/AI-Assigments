from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer
import torch

# Load and prepare dataset
dataset = load_dataset("json", data_files="../data/recipe_dataset.json")

# Use a small, efficient model (works on CPU)
model_name = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B parameters
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def format_data(example):
    """Format for instruction tuning"""
    text = f"### User: {example['question']}\n### Assistant: {example['answer']}"
    return {"text": text}

tokenized_dataset = dataset.map(
    lambda x: tokenizer(
        format_data(x)["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
)

# Load model (4-bit quantization for memory efficiency)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Reduces memory usage[citation:1]
    torch_dtype=torch.float32,
    device_map="auto"
)

# Prepare for LoRA training (trains only 1% of parameters)[citation:1]
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Training configuration
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=10,  # Adjust based on dataset size
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,  # Set to True if using GPU
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Train and save
trainer.train()
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
print("Model training complete!")
