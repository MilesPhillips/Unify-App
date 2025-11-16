# PowerShell script with examples of how to run LLM.py

Write-Host "=== LLM.py Usage Examples ===" -ForegroundColor Green

Write-Host "`n1. Chat with a pre-trained model (no training):" -ForegroundColor Yellow
Write-Host "python LLM.py --chat_only --model_name microsoft/DialoGPT-medium" -ForegroundColor Cyan

Write-Host "`n2. Chat with a local model:" -ForegroundColor Yellow
Write-Host "python LLM.py --chat_only --model_path ./results/final_model" -ForegroundColor Cyan

Write-Host "`n3. Train a model then chat:" -ForegroundColor Yellow
Write-Host "python LLM.py --model_name microsoft/DialoGPT-small --epochs 1 --batch_size 2" -ForegroundColor Cyan

Write-Host "`n4. Chat with custom settings:" -ForegroundColor Yellow
Write-Host "python LLM.py --chat_only --model_name microsoft/DialoGPT-medium --max_tokens 100 --output_file my_chat.jsonl" -ForegroundColor Cyan

Write-Host "`n5. Full training with evaluation:" -ForegroundColor Yellow
Write-Host "python LLM.py --model_name microsoft/DialoGPT-small --do_eval --epochs 2" -ForegroundColor Cyan

Write-Host "`n=== Available Arguments ===" -ForegroundColor Green
Write-Host "--chat_only          : Skip training, go directly to chat" -ForegroundColor White
Write-Host "--model_name         : HuggingFace model name (default: meta-llama/Meta-Llama-3-8B)" -ForegroundColor White
Write-Host "--model_path         : Path to local trained model" -ForegroundColor White
Write-Host "--max_tokens         : Max tokens to generate (default: 200)" -ForegroundColor White
Write-Host "--output_file        : File to save chat transcripts (default: llm_transcripts.jsonl)" -ForegroundColor White
Write-Host "--epochs             : Training epochs (default: 3)" -ForegroundColor White
Write-Host "--batch_size         : Batch size (default: 4)" -ForegroundColor White
Write-Host "--lr                 : Learning rate (default: 5e-5)" -ForegroundColor White
Write-Host "--do_eval            : Enable evaluation during training" -ForegroundColor White

Write-Host "`n=== Quick Start (Recommended) ===" -ForegroundColor Green
Write-Host "For a quick test with a smaller model:" -ForegroundColor Yellow
Write-Host "python LLM.py --chat_only --model_name microsoft/DialoGPT-medium" -ForegroundColor Cyan
