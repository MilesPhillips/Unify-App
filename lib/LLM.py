import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset
import argparse
import json

SYSTEM_INSTRUCTION = "You are a helpful assistant. Respond simply, clearly, and accurately."

class LLMChain:
    def __init__(self, model_name="gpt2", model_path=None, max_tokens=100, output_file="llm_transcripts.jsonl", system_instruction=SYSTEM_INSTRUCTION):
        self.model_name = model_name
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.output_file = output_file
        self.system_instruction = system_instruction
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        model_to_load = self.model_path if self.model_path else self.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
        except ImportError:
            print("Flash Attention not available, using standard attention...")
            model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.bfloat16,
                device_map="auto"
            )
        return model, tokenizer

    def build_prompt(self, history, user_msg):
        lines = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        lines.append(f"User: {user_msg}")
        lines.append("Assistant: (Respond simply and accurately)")
        return "\n".join(lines)

    def store_interaction(self, user_text, assistant_text):
        try:
            with open(self.output_file, "a") as f:
                f.write(json.dumps({"input": user_text, "output": assistant_text}) + "\n")
        except Exception as e:
            print(f"⚠️ Failed to write transcript: {e}")

    def generate_and_store_response(self, history, prompt):
        full_prompt = f"System: {self.system_instruction}\n" + self.build_prompt(history, prompt)
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )

        input_len = inputs['input_ids'].shape[1]
        gen_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        for stop_marker in ['\nUser:', '\nUser', 'User:', '\nYou:']:
            idx = response.find(stop_marker)
            if idx != -1:
                response = response[:idx].strip()
                break
        
        self.store_interaction(prompt, response)
        return response

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
    parser.add_argument("--system_instruction", default=SYSTEM_INSTRUCTION, help="System instruction prepended to every user prompt")
    return parser.parse_args()

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
        remove_columns=["text"]
    )
    return tokenized_dataset

def setup_training(model, tokenizer, args):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
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
        fp16=torch.cuda.is_available(),
        bf16=not torch.cuda.is_available(),
        warmup_steps=500,
        weight_decay=0.01,
        report_to="tensorboard",
        load_best_model_at_end=True if args.do_eval else False,
        optim="adamw_torch",
        max_grad_norm=1.0
    )
   
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"] if args.do_eval else None,
        data_collator=data_collator,
    )
    return trainer

if __name__ == "__main__":
    set_seed(42)
    args = setup_config()
    
    llm_chain = LLMChain(
        model_name=args.model_name,
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        output_file=args.output_file,
        system_instruction=args.system_instruction
    )
    
    if args.chat_only:
        history = []
        print(f"\n=== LLM Chat Interface ===")
        print(f"Model: {llm_chain.model_name if not llm_chain.model_path else llm_chain.model_path}")
        print(f"Max tokens: {llm_chain.max_tokens}")
        print(f"System Instruction: {llm_chain.system_instruction}")
        print(f"Transcripts will be saved to: {llm_chain.output_file}")
        print("Type your prompt. Type 'exit' to quit.\n")

        while True:
            prompt = input("You: ").strip()
            if prompt.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not prompt:
                continue
            
            response = llm_chain.generate_and_store_response(history, prompt)
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})
            print(f"LLM: {response}\n")
    else:
        print("Starting training pipeline...")
        
        tokenized_dataset = prepare_dataset(llm_chain.tokenizer, args.dataset_name, args.max_seq_length)
        
        trainer = setup_training(llm_chain.model, llm_chain.tokenizer, args)
        trainer.train()
        
        final_model_path = f"{args.output_dir}/final_model"
        trainer.save_model(final_model_path)
        llm_chain.tokenizer.save_pretrained(final_model_path)
        print(f"Model saved to: {final_model_path}")
        
        chat_prompt = input("\nWould you like to chat with the trained model? (y/N): ").strip().lower()
        if chat_prompt in ['y', 'yes']:
            history = []
            while True:
                prompt = input("You: ").strip()
                if prompt.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                if not prompt:
                    continue
                
                response = llm_chain.generate_and_store_response(history, prompt)
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": response})
                print(f"LLM: {response}\n")