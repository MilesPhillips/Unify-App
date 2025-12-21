# LLM.py Usage Guide

## Setup Complete! 🎉

Your LLM.py is now ready to use with terminal arguments. Here are the different ways you can launch and chat with your LLM:

## Quick Start Commands

### 1. **Chat Only Mode (No Training)**
```powershell
# Basic chat with GPT-2
python LLM.py --chat_only --model_name gpt2

# Chat with custom settings
python LLM.py --chat_only --model_name gpt2 --max_tokens 100 --output_file my_conversation.jsonl

# Chat with a different model
python LLM.py --chat_only --model_name microsoft/DialoGPT-medium
```

### 2. **Training + Chat Mode**
```powershell
# Train a model then chat
python LLM.py --model_name gpt2 --epochs 1 --batch_size 2

# Train with evaluation
python LLM.py --model_name gpt2 --do_eval --epochs 2
```

### 3. **Chat with Previously Trained Model**
```powershell
# Chat with a model you previously trained
python LLM.py --chat_only --model_path "./results/final_model"
```

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--chat_only` | Skip training, go directly to chat | False |
| `--model_name` | HuggingFace model name | meta-llama/Meta-Llama-3-8B |
| `--model_path` | Path to local trained model | None |
| `--max_tokens` | Max tokens to generate in chat | 200 |
| `--output_file` | File to save chat transcripts | llm_transcripts.jsonl |
| `--epochs` | Training epochs | 3 |
| `--batch_size` | Training batch size | 4 |
| `--lr` | Learning rate | 5e-5 |
| `--do_eval` | Enable evaluation during training | False |

## Example Workflows

### Beginner Workflow (Recommended)
1. Start with a small model for testing:
   ```powershell
   python LLM.py --chat_only --model_name gpt2 --max_tokens 50
   ```

2. Try a conversational model:
   ```powershell
   python LLM.py --chat_only --model_name microsoft/DialoGPT-small
   ```

### Advanced Workflow
1. Train your own model:
   ```powershell
   python LLM.py --model_name gpt2 --epochs 1 --batch_size 2 --do_eval
   ```

2. Chat with your trained model:
   ```powershell
   python LLM.py --chat_only --model_path "./results/final_model"
   ```

## Chat Commands
Once in chat mode:
- Type any message and press Enter to chat
- Type `exit` or `quit` to end the session
- All conversations are automatically saved to the specified output file

## Output Files
- **Chat transcripts**: Saved as JSON lines (.jsonl) format
- **Trained models**: Saved in `./results/final_model/` directory
- **Training logs**: Available in `./logs/` directory

## Troubleshooting

### Common Issues:
1. **Out of memory**: Use smaller `--batch_size` (e.g., 1 or 2)
2. **Model too slow**: Use smaller models like `gpt2` or `distilgpt2`
3. **Flash attention error**: Already handled automatically (fallback to standard attention)

### Performance Tips:
- For faster loading: Use smaller models (`gpt2`, `distilgpt2`)
- For better conversations: Use dialog models (`microsoft/DialoGPT-*`)
- For training: Start with `--epochs 1` and small `--batch_size`

## Testing Your Setup
Run the test script:
```powershell
.\test_llm_chat.ps1
```

This will start a chat session with GPT-2 and save the conversation to `test_chat.jsonl`.

## Ready to Go! 🚀
Your LLM is now fully configured for terminal-based chat. Try the basic command first:
```powershell
python LLM.py --chat_only --model_name gpt2 --max_tokens 50
```
