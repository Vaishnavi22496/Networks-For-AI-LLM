import torch
from torch import nn
from typing import Any, List, Sequence, Tuple, Union
import sys
sys.path.append('./tools')
import config as gemma_config
from model import GemmaDecoderLayer, RMSNorm
from network_utils import send_data, NumPySerializer, receive_data
import socket
import signal
import time
import numpy as np


class GemmaFirstLayerModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.first_layer = GemmaDecoderLayer(config)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.first_layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_cache, mask=mask)
        return hidden_states

class GemmaRemainingLayersModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(1, config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_caches[i], mask=mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
class GemmaLayerModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.layer = GemmaDecoderLayer(config)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_cache, mask=mask)
        return hidden_states
    
class GemmaLastLayerModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.layer = GemmaDecoderLayer(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_cache, mask=mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
class GemmaMiddleLayersModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(1, config.num_hidden_layers - 1)])
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_caches[i], mask=mask)
        return hidden_states
    
def load_model(model, filepath):
    """Loads a model's state dictionary from a specified filepath."""
    model.load_state_dict(torch.load(filepath))

def forward_pass(
    next_layer,
    server_socket,
    config,
    model,
    embedder,
    sampler,
    prec_freqs_cis,
    output_index,
    device,
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
    freqs_cis = prec_freqs_cis.index_select(0, input_positions)
    kv_write_indices = input_positions

    # [batch_size, input_len, hidden_size]
    hidden_states = embedder(input_token_ids)
    # Gemma normalizes the embedding by sqrt(hidden_size).
    hidden_states = hidden_states * (config.hidden_size**0.5)
    hidden_states = model[0](
        hidden_states=hidden_states,
        freqs_cis=freqs_cis,
        kv_write_indices=kv_write_indices,
        kv_cache=kv_caches[0],
        mask=mask,
    )
    # print(output_index.item())
    # print('Hidden states size: ', print_MB_size(hidden_states))
    # print('Freqs cis size: : ', print_MB_size(freqs_cis))
    # print('KV Write Indices size: : ', print_MB_size(kv_write_indices))
    # print('KV Caches size: ', print_MB_size(kv_caches[1][0], factor=len(kv_caches)*2))
    # print('Mask size: ', print_MB_size(mask))
    # Construct dictionary containing all the data
    data = {
        'hidden_state': hidden_states,
        'KV_index': kv_write_indices,
    }
    # Set up the socket
    # print("preparing to send data: {}".format(data))
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to the second layer
        # print("connecting to:".format(next_layer))
        client_socket.connect((next_layer['host'], next_layer['port']))
        # Send data to second layer
        send_data(client_socket, data)

    finally:
        # Close connection
        client_socket.close()

    # Receive response from last layer
    client_socket, client_addr = server_socket.accept()
    print(f"Accepted connection from {client_addr[0]}:{client_addr[1]}")
    deserialized_response = receive_data(client_socket)
    client_socket.close()
    
    # Extract tensor data from response
    hidden_states = deserialized_response['hidden_state']

    if len(model) == 2:
         hidden_states = model[1](
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_caches[1],
            mask=mask,
        )

    embedder_weight = embedder.weight
    if config.quant:
        embedder_weight = (
            embedder_weight * embedder.weight_scaler.unsqueeze(-1))
    next_tokens = sampler(
        embedding=embedder_weight,
        hidden_states=hidden_states.to(device), # After coming back from the server
        output_positions=output_positions,
        temperatures=temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
    )
    return next_tokens
        
def generation(
    prev_layer,
    next_layer,
    tokenizer,
    config,
    model,
    embedder,
    sampler,
    prec_freqs_cis,
    prompts: Union[str, Sequence[str]],
    device: Any,
    output_len: int = 100,
    filename=None,
    temperature: Union[float, None] = 0.95,
    top_p: float = 1.0,
    top_k: int = 100,
) -> Union[str, Sequence[str]]:
    """Generates responses for given prompts using Gemma model."""
    # If a single prompt is provided, treat it as a batch of 1.
    is_str_prompt = isinstance(prompts, str)
    if is_str_prompt:
        prompts = [prompts]

    batch_size = len(prompts)
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    # print(len(prompt_tokens[0]))
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    max_seq_len = max_prompt_len + output_len
    assert max_seq_len <= config.max_position_embeddings

    # build KV caches
    kv_caches = []
    for i in range(len(model)):
        size = (batch_size, max_seq_len, config.num_key_value_heads,
                config.head_dim)
        dtype = config.get_dtype()
        k_cache = torch.zeros(size=size, dtype=dtype, device=device)
        v_cache = torch.zeros(size=size, dtype=dtype, device=device)
        kv_caches.append((k_cache, v_cache))

    # prepare inputs
    token_ids_tensor = torch.full((batch_size, max_seq_len),
                                  tokenizer.pad_id, dtype=torch.int64)
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                        tokenizer.pad_id,
                                        dtype=torch.int64)
    
    # Set up the socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to the server
        client_socket.connect((next_layer['host'], next_layer['port']))
        # Necessary data to instantiate structures at the server side
        data = {'batch_size': batch_size,
                'max_seq_len': max_seq_len}
        # Send data to server
        send_data(client_socket, data)
    finally:
        # Close connection
        client_socket.close()

    for i, p in enumerate(prompt_tokens):
        token_ids_tensor[i, :len(p)] = torch.tensor(p)
        input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
            p[:min_prompt_len])
    token_ids_tensor = token_ids_tensor.to(device)
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    prompt_mask_tensor = token_ids_tensor != tokenizer.pad_id
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

    # Initialize server socket to listen for weights comming from last layer
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((prev_layer['host'], prev_layer['port']))
    server_socket.listen(5)
    print(f"Server listening on {prev_layer['host']}:{prev_layer['port']}...")

    def signal_handler(sig, frame):
        print('Closing server...')
        server_socket.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    for i in range(max_seq_len - min_prompt_len):
        if filename is not None:
            with open(filename, 'a') as file:
                file.write("Forward pass {} start: {:.6f}\n".format(i, time.time()))
        current_start = time.perf_counter()
        next_token_ids = forward_pass(
            next_layer,
            server_socket,
            config,
            model,
            embedder,
            sampler,
            prec_freqs_cis,
            output_index,
            device,
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
        print("Forward pass {} successful".format(i))
        current_end = time.perf_counter()
        execution_time = current_end - current_start
        print(f"Execution time: {execution_time:.6f} seconds")
        if filename is not None:
            with open(filename, 'a') as file:
                file.write("Forward pass {} execution time: {:.6f}\n".format(i, execution_time))
        
        curr_prompt_mask = prompt_mask_tensor.index_select(
            1, output_index).squeeze(dim=1)
        curr_token_ids = token_ids_tensor.index_select(
            1, output_index).squeeze(dim=1)
        output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                    next_token_ids).unsqueeze(dim=1)
        token_ids_tensor.index_copy_(1, output_index, output_token_ids)

        input_token_ids_tensor = output_token_ids
        input_positions_tensor = output_index.unsqueeze(dim=-1)
        curr_mask_tensor = mask_tensor.index_select(2,
                                                    input_positions_tensor)
        output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
            device)
        output_index = output_index + 1

    server_socket.close()
    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
        trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                + output_len]
        if tokenizer.eos_id in trimmed_output:
            eos_index = trimmed_output.index(tokenizer.eos_id)
            trimmed_output = trimmed_output[:eos_index]
        results.append(tokenizer.decode(trimmed_output))

    # If a string was provided as input, return a string as output.
    return results[0] if is_str_prompt else results

def dummy_generation(
    tokenizer,
    config,
    model,
    embedder,
    sampler,
    prec_freqs_cis,
    prompts: Union[str, Sequence[str]],
    device: Any,
    output_len: int = 100,
    temperature: Union[float, None] = 0.95,
    top_p: float = 1.0,
    top_k: int = 100):
    return "It's working"
