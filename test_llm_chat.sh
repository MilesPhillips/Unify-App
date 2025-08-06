#!/bin/bash
# Quick test script for LLM.py

echo "=== LLM.py Chat Test ==="
echo ""
echo "Starting LLM chat with GPT-2 model..."
echo "This will open an interactive chat session."
echo "Type 'exit' or 'quit' to end the session."
echo ""

# Run the LLM in chat mode
python LLM.py --chat_only --model_name gpt2 --max_tokens 50 --output_file test_chat.jsonl

echo ""
echo "Chat session ended. Check test_chat.jsonl for the conversation log."
