import os
import torch
from torch import nn
import argparse
import sys
# Add the tools directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.join(script_dir, 'tools')
sys.path.append(tools_dir)
from tools.config import get_config_for_7b, get_config_for_2b
from tools.model import precompute_freqs_cis
from tools.network_utils import send_data, receive_data
from tools.model_utils import GemmaLayerModel, GemmaLastLayerModel, load_model
import numpy as np
import socket
import signal
import time



def initialize_model(model_path, is_last_layer, config, device):
    """Return GemmaRemainingLayersModel with loaded weights"""
    layer_model = GemmaLastLayerModel(config) if is_last_layer else GemmaLayerModel(config)
    load_model(layer_model, model_path)
    layer_model.to(device)
    layer_model.eval()
    # Pre-compute rotary embedding table.
    rope_theta = getattr(config, 'rope_theta', 10000)
    prec_freqs_cis = precompute_freqs_cis(config.head_dim,
                                    config.max_position_embeddings * 2,
                                    theta=rope_theta).to(device)
    return layer_model, prec_freqs_cis

def initialize_gen_aux(mask_tensor, kv_caches, data):
    batch_size = data['batch_size']
    max_seq_len = data['max_seq_len']

    # build KV caches
    kv_caches = []
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


def handle_client(client_socket, model, prec_freqs_cis, mask_tensor, kv_caches, device, next_layer, is_last_layer):
    try:
        # Receive data from client
        received_data = receive_data(client_socket)
        if 'batch_size' in received_data.keys():
            # print("Received batch size data: {}".format(received_data))
            # This means it is time to initialize generation structures
            mask_tensor, kv_caches = initialize_gen_aux(mask_tensor, kv_caches, received_data)
            # print("initialized mask_tensor: {}".format(mask_tensor))
            if not is_last_layer:
                # print("Not the last layer, passing info on")
                # We need to pass the received information forward
                next_layer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    next_layer_socket.connect((next_layer['host'], next_layer['port']))
                    send_data(next_layer_socket, received_data)
                finally:
                    next_layer_socket.close()
            return (mask_tensor, kv_caches)
        else:
            # print("Performing forward pass")
            # print("Received data: {}".format(received_data))
            # print("input positions: {}".format(received_data['KV_index']))
            input_positions = received_data['KV_index'].to(device)
            freqs_cis = prec_freqs_cis.index_select(0, input_positions)
            # print("freqs_cis: {}".format(freqs_cis))
            curr_mask_tensor = mask_tensor.index_select(2, input_positions)

            hidden_states = model(
                hidden_states=received_data['hidden_state'].to(device),
                freqs_cis=freqs_cis,
                kv_write_indices=input_positions,
                kv_cache=kv_caches[0],
                mask=curr_mask_tensor,
            )
            # Prepare message to next layer
            data = {
                'hidden_state': hidden_states,
                'KV_index': input_positions,
            }
            # print("creating connection to next layer")
            next_layer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                next_layer_socket.connect((next_layer['host'], next_layer['port']))
                # print("connected to next layer")
                send_data(next_layer_socket, data)
                # print("data sent to next layer")
            except ConnectionRefusedError as e:
                print(f"Error: Unable to connect to {next_layer['host']}:{next_layer['port']}")
                print(e)
                # Implement a retry mechanism
                retry_count = 5
                for attempt in range(retry_count):
                    try:
                        time.sleep(1)  # Wait before retrying
                        next_layer_socket.connect((next_layer['host'], next_layer['port']))
                        print(f"Successfully connected on attempt {attempt + 1}")
                        send_data(next_layer_socket, data)
                        break
                    except ConnectionRefusedError as retry_e:
                        print(f"Retry {attempt + 1}/{retry_count} failed: {retry_e}")
                        if attempt == retry_count - 1:
                            print("All retry attempts failed. Closing client connection.")
                            client_socket.close()
                            return 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                return 1
            finally:
                # print("closing connection to next layer")
                next_layer_socket.close()
    finally:
        # print("closing client socket")
        client_socket.close()

    return 0


def start_server(prev_layer, next_layer, is_last_layer, filename):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((prev_layer['host'], prev_layer['port']))
    server_socket.listen()
    # print("Server listening on {}:{}...".format(prev_layer['host'], prev_layer['port']))
    model, prec_freqs_cis = initialize_model(MODEL_PATH, is_last_layer, config, device)
    mask_tensor, kv_caches = None, None
    k = 0
    # print("Model ready for inference")

    def signal_handler(sig, frame):
        print('Closing server...')
        server_socket.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            client_socket, client_addr = server_socket.accept()
            with open(filename, 'a') as file:
                    file.write("Accepted Connection: {:.6f}\n".format(time.time()))
                    accept_time = time.perf_counter()
            # print("Accepted connection from {}:{}".format(client_addr[0], client_addr[1]))
            response = handle_client(client_socket, model, prec_freqs_cis, mask_tensor, kv_caches, device, next_layer, is_last_layer)
            if not isinstance(response, int):
                # print("intialiaze generation")
                # If server was called to initialize generation
                mask_tensor, kv_caches = response
                with open(filename, 'a') as file:
                    file.write("Start time: {:.6f}\n".format(time.time()))
                k = 0 # To track token generated
                # print("mask_tensor {}".format(mask_tensor))
            # else:
                # print("normal operation")
                # print("response: {}".format(response))
            else:
                with open(filename, 'a') as file:
                    file.write("Forward pass {}: {:.6f}\n".format(k, time.perf_counter() - accept_time))
            k += 1
    finally:
        server_socket.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Introduce host and port of previous and next node.")
    parser.add_argument("--host_previous", type=str, default='127.0.0.1', help="Host IP from previous layer")
    parser.add_argument("--port_previous", type=int, default=12346, help="Port from previous layer")
    parser.add_argument("--host_next", type=str, default='127.0.0.1', help="Host IP from next layer")
    parser.add_argument("--port_next", type=int, default=12346, help="Port from next layer")
    parser.add_argument("--layer", type=int, default=1, help="Layer number (starting from 0 to num_hidden_layers - 1)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prev_layer = {'host': args.host_previous, 'port': args.port_previous}
    next_layer = {'host': args.host_next, 'port': args.port_next}
    i = args.layer
    # Define model variables
    # Choose variant and machine type
    VARIANT = '1.1-2b-it'
    MACHINE_TYPE = 'cuda'
    MODEL_PATH = './weights/{}/layer_model_{}.pth'.format(VARIANT, i)
    LOGS_DIR = '/logs'

    # Create the directory if it doesn't exist
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        print('/logs directory not found')
    # Create file
    filename = os.path.join(LOGS_DIR, 'timing_layer_{}.txt'.format(i))
    with open(filename, 'a'):  # 'a' mode creates the file if it doesn't exist and appends to it
        pass  # Do nothing inside the context manager, as the file is already created
    # Define config
    config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    config.quant = 'quant' in VARIANT
    torch.set_default_dtype(config.get_dtype())
    device = torch.device('cuda' if torch.cuda.is_available() and MACHINE_TYPE == 'cuda' else 'cpu')
    is_last_layer = i == config.num_hidden_layers-1
    start_server(prev_layer, next_layer, is_last_layer, filename)
