
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMService:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initializes the LLM service by loading the model and tokenizer.
        """
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_name)
        self.device = self.model.device
        print(f"LLMService initialized with model {model_name} on device {self.device}")

    def _load_model_and_tokenizer(self, model_name):
        """
        Loads a HuggingFace model and tokenizer.
        Private method for internal use.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Determine the torch dtype based on CUDA availability
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"Using dtype: {dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        
        model.eval()
        return model, tokenizer

    def build_prompt(self, history, system_instruction):
        """
        Builds a prompt string for the model from conversation history.
        
        Args:
            history (list): A list of dictionaries, e.g., [{"role": "user", "content": "..."}]
            system_instruction (str): The system instruction for the model.

        Returns:
            str: The fully formatted prompt.
        """
        lines = [f"System: {system_instruction}"]
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def generate_response(self, prompt, max_new_tokens=150, temperature=0.7, top_p=0.9):
        """
        Generates a response from the loaded model.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )

        # Decode and clean up the response
        input_len = inputs['input_ids'].shape[1]
        gen_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Truncate at stop markers to prevent the model from generating for the user
        for stop_marker in ['\nUser:', '\nSystem:', '</s>']:
            idx = response.find(stop_marker)
            if idx != -1:
                response = response[:idx].strip()
        
        return response
