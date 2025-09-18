# python LLM.py --chat_only --model_name gpt2 --max_tokens 100
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    pipeline
)
#from datasets import load_dataset
import argparse
import json


def setup_config():
    parser = argparse.ArgumentParser(description="LLM Training and Chat Interface")
    parser.add_argument("--model_name", default="gpt2", help="HuggingFace model name or path")
    parser.add_argument("--model_path", default=None, help="Local model path to load for chat mode")
    parser.add_argument("--chat_only", action="store_true", help="Skip training and go directly to chat mode")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate in chat mode")
    parser.add_argument("--output_file", default="llm_transcripts.jsonl", help="File to save chat transcripts")
    parser.add_argument("--dataset_name", default="your_dataset", help="HuggingFace dataset or local path for training")
    parser.add_argument("--output_dir", default="./results", help="Output directory for checkpoints")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation during training")
    parser.add_argument("--system_instruction", default="You are a friend of the user.", help="System instruction prepended to every user prompt")
    return parser.parse_args()

def build_prompt(history, user_msg):
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    lines.append(f"User: {user_msg}")
    lines.append("Assistant:")
    return "\n".join(lines)

def store_interaction(user_text, assistant_text, path="llm_transcripts.jsonl"):
    try:
        with open(path, "a") as f:
            f.write(json.dumps({"input": user_text, "output": assistant_text}) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to write transcript: {e}")


# 2. Prepare Model and Tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    try:
        # Try with flash attention first
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Better for modern GPUs
            device_map="auto",           # Auto-distribute across GPUs
            #attn_implementation="flash_attention_2"  # If available
        )
    except ImportError:
        # Fallback without flash attention (for CPU or systems without flash_attn)
        print("Flash Attention not available, using standard attention...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.bfloat16,
            device_map="auto"
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

def llm_generate_response(transcript, pipe, max_tokens=200):
    model.eval()
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    messages = [
        {"role": "system", "content": "You are a caring patient friend."},
    ]
    messages[1]["content"] = transcript
    #prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    output = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(output[0]["generated_text"])

    return output


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
    
def generate_and_store_response(model, tokenizer, args):
    """Interactive chat function with the trained model"""
    model.eval()
    device = model.device
    
    history = []

    # System instruction - This is where the LLM gets its instructions!
    system_instruction = args.system_instruction

    print(f"\n=== LLM Chat Interface ===")
    print(f"Model: {args.model_name if not args.model_path else args.model_path}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"System Instruction: {system_instruction}")
    print(f"Transcripts will be saved to: {args.output_file}")
    print("Type your prompt. Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not prompt:
            continue
        
          # Add user message to history
        history.append({"role": "user", "content": prompt})

        # Combine system instruction with user prompt
        full_prompt = f"System: {system_instruction}\n" + build_prompt(history[:-1], prompt)
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )

        # Only get the new generated tokens (exclude the input prompt)
        input_len = inputs['input_ids'].shape[1]
        gen_tokens = outputs[0][input_len:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Truncate at the next user turn if the model emits it (avoid cross-talk)
        for stop_marker in ['\nUser:', '\nUser', 'User:', '\nYou:']:
            idx = response.find(stop_marker)
            if idx != -1:
                response = response[:idx].strip()
                break
        print(f"LLM: {response}\n")

        # Store the conversation (include system instruction for context)
        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "system_instruction": system_instruction,
                "input": prompt,
                "output": response
            }) + "\n")

# 5. Main Execution
if __name__ == "__main__":
    set_seed(42)  # For reproducibility
    args = setup_config()
    
    print(f"=== LLM Setup ===")
    print(f"Chat only mode: {args.chat_only}")
    print(f"Model: {args.model_name if not args.model_path else args.model_path}")
    
    if args.chat_only:
        # Load model for chat only (no training)
        model_to_load = args.model_path if args.model_path else args.model_name
        print(f"Loading model for chat: {model_to_load}")
        model, tokenizer = load_model_and_tokenizer(model_to_load)
        generate_and_store_response(model, tokenizer, args)
    else:
        # Full training pipeline
        print("Starting training pipeline...")
        
        # Load components
        model, tokenizer = load_model_and_tokenizer(args.model_name)
        tokenized_dataset = prepare_dataset(tokenizer, args.dataset_name, args.max_seq_length)
        
        # Train
        trainer = setup_training(model, tokenizer, args)
        trainer.train()
        
        # Save final model
        final_model_path = f"{args.output_dir}/final_model"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Model saved to: {final_model_path}")
        
        # Ask if user wants to chat with the trained model
        chat_prompt = input("\nWould you like to chat with the trained model? (y/N): ").strip().lower()
        if chat_prompt in ['y', 'yes']:
            generate_and_store_response(model, tokenizer, args)