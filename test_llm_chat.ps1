# Quick test script for LLM.py (PowerShell)

Write-Host "=== LLM.py Chat Test ===" -ForegroundColor Green
Write-Host ""
Write-Host "Starting LLM chat with GPT-2 model..." -ForegroundColor Yellow
Write-Host "This will open an interactive chat session." -ForegroundColor White
Write-Host "Type 'exit' or 'quit' to end the session." -ForegroundColor White
Write-Host ""

# Run the LLM in chat mode
python lib/LLM.py --chat_only --model_name gpt2 --max_tokens 50 --output_file test_chat.jsonl

Write-Host ""
Write-Host "Chat session ended. Check test_chat.jsonl for the conversation log." -ForegroundColor Green
