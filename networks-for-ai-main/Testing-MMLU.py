import os
import re
import argparse
import kagglehub
from tools.config import get_config_for_7b, get_config_for_2b
import tools.config as gemma_config
from tools.model import Sampler, Embedding, precompute_freqs_cis
from tools.model_utils import GemmaLayerModel, GemmaLastLayerModel, load_model
from tools.tokenizer import Tokenizer
import torch
from torch import nn
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.models.base_model import DeepEvalBaseLLM
from typing import Any, List, Optional, Sequence, Tuple, Union

# Machine type
VARIANT = '1.1-2b-it'
MACHINE_TYPE = 'cuda'

def save_results(results, file_path, layers=None):  
    # Open the file in append mode and write the results
    with open(file_path, "a") as file:
        if layers is not None:
            if isinstance(layers, int):
                file.write(f"{layers}-{layers}:")
            else:
                file.write(f"{layers[0]}-{layers[1]}:")
        file.write(str(results))
        file.write("\n")  # Separator
        
def rename_file_if_exists(directory, filename):
    # Construct the initial file path
    file_path = os.path.join(directory, filename)
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Extract the base name and extension
        base_name, extension = os.path.splitext(filename)
        
        # Start with the first increment
        i = 1
        
        # Construct new file name with incrementing integer until a non-existent file name is found
        while os.path.exists(os.path.join(directory, f"{base_name}{i}{extension}")):
            i += 1
        
        # Construct the new file path
        new_file_path = os.path.join(directory, f"{base_name}{i}{extension}")
        print(f"File renamed to: {new_file_path}")
        file_path = new_file_path
    return file_path

class GemmaForCausalLM(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        device
    ):
        super().__init__()
        self.config = config
        self.device = device
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = Tokenizer(config.tokenizer)
        self.embedder = self.initialize_embedder(vocab_size)
        self.model = self.initialize_model()
        self.sampler = Sampler(vocab_size).to(self.device)

        # Pre-compute rotary embedding table.
        rope_theta = getattr(config, 'rope_theta', 10000)
        self.freqs_cis = precompute_freqs_cis(head_dim,
                                         max_seq_len * 2,
                                         theta=rope_theta).to(self.device)
        
    def initialize_model(
        self,
    ):
        # Initialize and load splits
        model = [GemmaLayerModel(self.config) for layer in range(self.config.num_hidden_layers - 1)]
        model.append(GemmaLastLayerModel(self.config))
        for layer in range(self.config.num_hidden_layers):
            load_model(model[layer], f'./weights/{VARIANT}/layer_model_{layer}.pth')
        model = [layer.to(self.device).eval() for layer in model]
        return model

    def initialize_embedder(
        self,
        vocab_size
    ):
        # Initialize embedder
        embedder = Embedding(vocab_size, self.config.hidden_size, self.config.quant).to(self.device)
        embedding_weights = torch.load(f'./weights/{VARIANT}/embedding_weights.pth', map_location=self.device)
        embedder.load_state_dict(embedding_weights)
        return embedder

    def forward_pass(
        self,
        output_index,
        opt_ixs,
        j, # Layer to skip if not none
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        freqs_cis = self.freqs_cis.index_select(0, input_positions)
        kv_write_indices = input_positions

        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        hidden_states = hidden_states * (self.config.hidden_size**0.5)
        with torch.no_grad():
            for i in range(self.config.num_hidden_layers):
                if j is not None and i in j:
                    #print(f'Skipped layer {i}')
                    continue
                hidden_states = self.model[i](
                    hidden_states=hidden_states,
                    freqs_cis=freqs_cis,
                    kv_write_indices=kv_write_indices,
                    kv_cache=kv_caches[i],
                    mask=mask,
                )
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = (
                embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
        
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedder_weight.t())
        # Selecting only certain indices from the logits tensor
        selected_logits = logits[:, opt_ixs]
        return torch.argmax(selected_logits, dim=-1).squeeze(dim=-1)
        #next_tokens = self.sampler(
        #    embedding=embedder_weight,
        #    hidden_states=hidden_states,
        #    output_positions=output_positions,
        #    temperatures=temperatures,
        #    top_ps=top_ps,
        #    top_ks=top_ks,
        #)
        #return next_tokens

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        lts: Union[int, Sequence[int], None],
        output_len: int = 1,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> Union[str, Sequence[str]]:
        """Generates responses for given prompts using Gemma model."""
        options = ['A', 'B', 'C', 'D']
        opt_ixs = [self.tokenizer.encode(letter)[1] for letter in options]
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]
        is_lts_int = isinstance(lts, int)
        if is_lts_int:
            lts = [lts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        # build KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                    self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                      self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_id,
                                            dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                              dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
            device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        next_token_ids = self.forward_pass(
            output_index,
            opt_ixs,
            lts,
            input_token_ids=input_token_ids_tensor,
            input_positions=input_positions_tensor,
            kv_write_indices=None,
            kv_caches=kv_caches,
            mask=curr_mask_tensor,
            output_positions=output_positions_tensor,
            temperatures=temperatures_tensor,
            top_ps=top_ps_tensor,
            top_ks=top_ks_tensor,
        )
        # Detokenization.
        token_ids = opt_ixs[next_token_ids.tolist()]
        result = self.tokenizer.decode(token_ids)

        # If a string was provided as input, return a string as output.
        return result



class GemmaEval(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        config,
        device,
        layer_to_skip=None
    ):
        self.model = model
        self.variant = VARIANT
        self.config = config
        self.device = device
        # Chat templates
        self.MODEL_PROMPT = '<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n'
        self.lts = layer_to_skip

    def load_model(self):
        return self.model
    
    def format_prompt(self, text):
        # Split the text into sections based on question starts
        questions = text.split('\n\n')
        # This will hold the final formatted text
        prepend = "Please read the following questions carefully and select the best answer on the last question based on logical reasoning and factual knowledge. Your goal is to identify the most accurate and scientifically valid response. "
        formatted_text = [prepend + questions[0]]

        # We use regex to capture the questions and their answers
        pattern = r"(.*?Answer: )([ABCD])(\n|$)"
        # Process each section
        for section in questions[1:-1]:
            # Find all the parts that need to be modified
            # This finds all questions and their answers
            matches = re.findall(pattern, section, re.DOTALL)
            # We process all matches except the last one as regular Q&A
            question, answer, end = matches[0]
            replacement = f"<start_of_turn>user\n{question}<end_of_turn><start_of_turn>model\n{answer}<end_of_turn>"
            #formatted_section = formatted_section.replace(question + 'Answer: ' + answer + end, replacement)

            # Append the formatted section to the result
            formatted_text.append(replacement)
        # For the last question, we do not need the model's answer part
        last_question = questions[-1]
        pattern = r"^(.*?Answer:)(.*)$"
        matches = re.findall(pattern, last_question, re.DOTALL)
        # We process all matches except the last one as regular Q&A
        question, end = matches[0]
        last_replacement = f"<start_of_turn>user\n{question} <end_of_turn>"
        formatted_text.append(last_replacement)
        return "\n\n".join(formatted_text)
    
    def generate(self, prompt: str) -> str:
        model = self.load_model()
        device = self.device
        #print(prompt)
        prompt = self.format_prompt(prompt)
        response = model.generate(
            prompts=self.MODEL_PROMPT.format(prompt=prompt),  # use the keyword for prompts
            device=device,
            lts = self.lts,
            output_len=1
        )
        return response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Gemma " + self.variant

def main():
    # Download tokenizer
    home_dir = os.path.expanduser("~")
    weights_dir = os.path.join(home_dir, ".cache", "kagglehub", "models", "google", "gemma", "pyTorch", VARIANT)
    weights_dir = os.path.join(weights_dir, "1") if '1.1' in VARIANT else os.path.join(weights_dir, "2")
    if not os.path.exists(weights_dir):
        kagglehub.login() # API KEY: 5cb66339276d4bea7ba59ca714d28f6b
        weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')
    tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
    assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'
    # Define config
    config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    config.tokenizer = tokenizer_path
    config.quant = 'quant' in VARIANT
    torch.set_default_dtype(config.get_dtype())
    device = torch.device('cuda' if torch.cuda.is_available() and MACHINE_TYPE == 'cuda' else 'cpu')
    
    gemma2B = GemmaForCausalLM(config, device)
    # Define the directory and file path
    directory = "results"
    file_path = rename_file_if_exists(directory, "resultsMMLU.txt") #os.path.join(directory, "resultsMMLU3.txt")
    benchmark = MMLU(
        #tasks=[MMLUTask.ASTRONOMY, 
        #       MMLUTask.CLINICAL_KNOWLEDGE,
        #       MMLUTask.COLLEGE_BIOLOGY,
        #       MMLUTask.COLLEGE_CHEMISTRY,
        #       MMLUTask.COLLEGE_COMPUTER_SCIENCE,
        #       MMLUTask.COLLEGE_MATHEMATICS,
        #       MMLUTask.COLLEGE_MEDICINE,
        #       MMLUTask.COLLEGE_PHYSICS,
        #       MMLUTask.COMPUTER_SECURITY,
        #       MMLUTask.CONCEPTUAL_PHYSICS,
        #       MMLUTask.ECONOMETRICS,
        #       MMLUTask.ELECTRICAL_ENGINEERING,
        #       MMLUTask.ABSTRACT_ALGEBRA,
        #       MMLUTask.MACHINE_LEARNING,
        #       MMLUTask.MISCELLANEOUS],
        n_shots=5
    )
    # Not skipping any layer
    gemma_2b = GemmaEval(model=gemma2B, config=config, device=device)
    results = benchmark.evaluate(model=gemma_2b)
    print("Overall Score: ", results)
    save_results(results, file_path)

    # Skipping one at a time
    for i in range(config.num_hidden_layers):
        gemma_2b = GemmaEval(model=gemma2B, config=config, device=device, layer_to_skip=i)
        results = benchmark.evaluate(model=gemma_2b)
        print("Overall Score: ", results)
        save_results(results, file_path, layers=i)

    # Skipping two layers at a time
    for i in range(config.num_hidden_layers):
        for j in range(i+1, config.num_hidden_layers):
            gemma_2b = GemmaEval(model=gemma2B, config=config, device=device, layer_to_skip=[i,j])
            results = benchmark.evaluate(model=gemma_2b)
            print("Overall Score: ", results)
            save_results(results, file_path, layers=[i,j])

if __name__ == "__main__":
    main()