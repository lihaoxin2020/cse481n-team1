from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    OPTForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
    T5ForConditionalGeneration
)
import torch.nn.functional as F
import torch
import sys

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

opt_iml_dic = {
    "opt-iml-1.3b": "facebook/opt-iml-1.3b",
    "opt-iml-30b": "facebook/opt-iml-30b"
}

t5_dic = {
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "t5-3b": "t5-3b",
    "t5-11b": "t5-11b"
}

flan_t5_dic = {
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
    "flan-t5-xxl": "google/flan-t5-xxl"
}


class Engine:
    def __init__(self, model_name):
        if model_name.startswith("alpaca"):
            self.engine = model_name
            self.tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
            self.model = LlamaForCausalLM.from_pretrained(
                "chainyo/alpaca-lora-7b",
                load_in_8bit=True,
                cache_dir="./.cache",
                torch_dtype=torch.float16,
                device_map="auto",
            )

        if model_name.startswith("t5"):
            self.engine = model_name
            self.tokenizer = T5Tokenizer.from_pretrained(t5_dic[model_name])
            self.model = T5ForConditionalGeneration.from_pretrained(t5_dic[model_name]).to(device)

        elif model_name.startswith("flan-t5"):
            self.engine = model_name
            self.tokenizer = T5Tokenizer.from_pretrained(flan_t5_dic[model_name], truncation=True,
                                                         model_max_length=5642)
            self.model = T5ForConditionalGeneration.from_pretrained(flan_t5_dic[model_name]).to(device)
        
        elif model_name.startswith("opt-iml"):
            self.engine = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(opt_iml_dic[model_name], use_fast = False)
            self.model = AutoModelForCausalLM.from_pretrained(opt_iml_dic[model_name]).to(device)

        elif model_name.startswith("opt"):
            self.engine = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(opt_dic[model_name], truncation=True, model_max_length=3189)
            self.model = OPTForCausalLM.from_pretrained(opt_dic[model_name]).to(device)
        
        else:
            print("model not recognised")
            

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        self.model.eval()

    def complete(self, prompt, max_tokens=64):

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=max_tokens, num_return_sequences=1)

        # Decode the generated tokens into text
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if 't5' in self.engine:
            completed_text = output_text
        else:
            completed_text = output_text[len(prompt):].lstrip()

        return completed_text

    def check_prompt_length(self, prompt, max_tokens=64):

        if (self.engine.startswith("t5-small")):
            prompt_length = len(self.tokenizer.encode(prompt))
            if prompt_length + max_tokens >= 3189:  # Prompt is too long
                return True
            return False

        prompt_length = len(self.tokenizer.encode(prompt))
        if prompt_length + max_tokens >= 2048:  # Prompt is too long
            return True
        return False

    def get_prob(self, prompt, num_tokens):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute the sum of the log-probabilities from the num_tokens token to the end
        partial_log_likelihood = log_probs[:, -num_tokens:, :].gather(-1, input_ids[:, -num_tokens:].unsqueeze(-1))

        return partial_log_likelihood.sum().item()
