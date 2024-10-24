import os
import torch
from torch import nn
import argparse
from tools.config import get_config_for_7b, get_config_for_2b
from tools.model import GemmaDecoderLayer, RMSNorm, Sampler, Embedding, precompute_freqs_cis
from tools.network_utils import send_data, receive_data
from tools.model_utils import GemmaRemainingLayersModel, GemmaMiddleLayersModel, load_model
import numpy as np
import socket
import signal
import sys


def initialize_model(model_path, config, device):
    """Return GemmaRemainingLayersModel with loaded weights"""
    middle = 'middle' in model_path
    remaining_layers_model = GemmaMiddleLayersModel(config) if middle else GemmaRemainingLayersModel(config)
    load_model(remaining_layers_model, model_path)
    remaining_layers_model.to(device)
    remaining_layers_model.eval()
    # Pre-compute rotary embedding table.
    rope_theta = getattr(config, 'rope_theta', 10000)
    prec_freqs_cis = precompute_freqs_cis(config.head_dim,
                                    config.max_position_embeddings * 2,
                                    theta=rope_theta).to(device)
    return remaining_layers_model, prec_freqs_cis

def initialize_gen_aux(mask_tensor, kv_caches, data):
    batch_size = data['batch_size']
    max_seq_len = data['max_seq_len']

    # build KV caches
    kv_caches = []
    for _ in range(1, config.num_hidden_layers):
        size = (batch_size, max_seq_len, config.num_key_value_heads,
                config.head_dim)
        dtype = config.get_dtype()
        k_cache = torch.zeros(size=size, dtype=dtype, device=device)
        v_cache = torch.zeros(size=size, dtype=dtype, device=device)
        kv_caches.append((k_cache, v_cache))

    # build mask tensor
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                             -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    return mask_tensor, kv_caches

def handle_client(client_socket, model, prec_freqs_cis, mask_tensor, kv_caches, device):
    # Receive data from client
    received_data = receive_data(client_socket)
    if 'batch_size' in received_data.keys():
        # This means it is time to initalize generation structures
        mask_tensor, kv_caches = initialize_gen_aux(mask_tensor, kv_caches, received_data)
        # Close client connection
        client_socket.close()
        return (mask_tensor, kv_caches)
    else:
        input_positions = received_data['KV_index'].to(device)
        freqs_cis = prec_freqs_cis.index_select(0, input_positions)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions)
        # Print each key-value pair
        # for key, value in received_data.items():
        #     print(f"{key}: {value}")
        # Dummy response: hidden_states = np.random.randint(1, 100, size=(100,), dtype=np.int64)
        # Generate response hidden states by passing received data through model
        hidden_states = model(
            hidden_states=received_data['hidden_state'].to(device),
            freqs_cis=freqs_cis,
            kv_write_indices=input_positions,
            kv_caches=kv_caches,
            mask=curr_mask_tensor,
        )
        # Prepare response
        response = {'hidden_state': hidden_states}
        # print('Response message: ', response)

        # Send response back to client using existing connection
        send_data(client_socket, response)

    # Close client connection
    client_socket.close()
    return 0


def start_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}...")
    model, prec_freqs_cis = initialize_model(MODEL_PATH, config, device)
    mask_tensor, kv_caches = None, None
    print("Model ready for inference")

    def signal_handler(sig, frame):
        print('Closing server...')
        server_socket.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            client_socket, client_addr = server_socket.accept()
            print(f"Accepted connection from {client_addr[0]}:{client_addr[1]}")
            response = handle_client(client_socket, model, prec_freqs_cis, mask_tensor, kv_caches, device)
            if not isinstance(response, int):
                # If server was called to initialize generation
                mask_tensor, kv_caches = response
    finally:
        server_socket.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Introduce host and port of previous and next node.")
    parser.add_argument("--host", type=str, default='127.0.0.1', help="Host IP from previous layer")
    parser.add_argument("--port", type=int, default=12346, help="Port from previous layer")
    parser.add_argument("--middle_layers", action="store_true", default=False, help="Use only middle layers, not all remaining decoder layers")

if __name__ == "__main__":
    args = parse_args()
    host = args.host
    port = args.port
    middle = args.middle_layers
    # Define model variables
    # Choose variant and machine type
    VARIANT = '2b-it'
    MACHINE_TYPE = 'cuda'
    MODEL_PATH = f'./weights/{VARIANT}/middle_layers_model.pth' if middle else f'./weights/{VARIANT}/remaining_layers_model.pth'
    # Define config
    config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    config.quant = 'quant' in VARIANT
    torch.set_default_dtype(config.get_dtype())
    device = torch.device('cuda' if torch.cuda.is_available() and MACHINE_TYPE == 'cuda' else 'cpu')
    start_server(host, port)
