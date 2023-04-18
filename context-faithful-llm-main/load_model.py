from transformers import (
    AutoTokenizer, 
    OPTForCausalLM, 
    LlamaTokenizer,
    LlamaForCausalLM,
    )
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

length_limit = {
    'text-davinci-003': 4096,
    'text-curie-001': 2048,
    'text-babbage-001': 2048,
    'text-ada-001': 2048,
}

opt_dic = {
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-13b": "facebook/opt-13b",
    "opt-30b": "facebook/opt-30b",
    "opt-66b": "facebook/opt-66b"
}

class Engine:
    def __init__(self, model_name):
        if model_name.startswith("alpaca"):
            self.engine = model_name
            self.tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
            self.model = LlamaForCausalLM.from_pretrained(
                "chainyo/alpaca-lora-7b",
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
            ).to(device)

        if model_name.startswith("opt"):
            self.engine = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(opt_dic[model_name])
            self.model = OPTForCausalLM.from_pretrained(opt_dic[model_name]).to(device)

        if torch.__version__ >= "2":
            self.model = torch.compile(self.model)
        self.model.eval()

    def check_prompt_length(self, prompt, max_tokens=64):
        prompt_length = len(self.tokenizer.encode(prompt))
        if prompt_length + max_tokens >= 2048:  # Prompt is too long
            return True
        return False

    def complete(self, prompt, max_tokens=64):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=max_tokens, num_return_sequences=1)

        # Decode the generated tokens into text
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completed_text = output_text[len(prompt):].lstrip()
        return completed_text

    def get_prob(self, prompt, num_tokens):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute the sum of the log-probabilities from the num_tokens token to the end
        partial_log_likelihood = log_probs[0, num_tokens:-1, :].gather(1, input_ids[:, num_tokens + 1:].unsqueeze(-1)).sum().item()

        return partial_log_likelihood
