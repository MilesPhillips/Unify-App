import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_dataset
import argparse
import json


# 1. Configuration
def setup_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B",
                      help="Base model to fine-tune")
    parser.add_argument("--dataset_name", default="your_dataset",
                      help="HuggingFace dataset or path to local data")
    parser.add_argument("--output_dir", default="./results",
                      help="Output directory for checkpoints")
    parser.add_argument("--lr", type=float, default=5e-5,
                      help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Per device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                      help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=512,
                      help="Maximum sequence length")
    return parser.parse_args()

# 2. Prepare Model and Tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
   
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Better for modern GPUs
        device_map="auto",           # Auto-distribute across GPUs
        attn_implementation="flash_attention_2"  # If available
    )
    return model, tokenizer

# 3. Dataset Preparation
def prepare_dataset(tokenizer, dataset_name, max_length):
    dataset = load_dataset(dataset_name)
   
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
   
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]  # Remove raw text column
    )
    return tokenized_dataset

# 4. Training Setup
def setup_training(model, tokenizer, args):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )
   
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if args.do_eval else "no",
        fp16=torch.cuda.is_available(),  # Enable mixed precision
        bf16=not torch.cuda.is_available(),  # Fallback to bfloat16
        warmup_steps=500,
        weight_decay=0.01,
        report_to="tensorboard",
        load_best_model_at_end=True if args.do_eval else False,
        optim="adamw_torch",
        max_grad_norm=1.0  # Gradient clipping
    )
   
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"] if args.do_eval else None,
        data_collator=data_collator,
    )
    return trainer

# 5. Main Execution
if __name__ == "__main__":
    set_seed(42)  # For reproducibility
    args = setup_config()
   
    # Load components
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    tokenized_dataset = prepare_dataset(tokenizer, args.dataset_name, args.max_seq_length)
   
    # Train
    trainer = setup_training(model, tokenizer, args)
    trainer.train()
   
    # Save final model
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
def generate_and_store_response(model, tokenizer, output_file="llm_transcripts.jsonl", max_tokens=200):
    model.eval()
    device = model.device

    print("\nType your prompt. Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            break

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"LLM: {response}\n")

        with open(output_file, "a") as f:
            f.write(json.dumps({"input": prompt, "output": response}) + "\n")
