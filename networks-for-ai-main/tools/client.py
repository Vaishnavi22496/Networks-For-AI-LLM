import numpy as np
from network_utils import send_data, receive_data
import socket
import torch

import torch
import numpy as np


def generate_random_data():
    # Generate random floating-point tensors for each message type
    hidden_state = torch.rand(1, 7, 10)
    CIS = torch.rand(2)
    KV_index = torch.rand(2)
    KV_cache = [torch.rand(2) for _ in range(5)]  # List of random tensors
    mask = torch.rand(2)

    # Construct dictionary containing all the random tensors
    data = {
        'hidden_state': hidden_state,
        'CIS': CIS,
        'KV_index': KV_index,
        'KV_cache': KV_cache,
        'mask': mask
    }

    return data


def main():
    # Server details
    HOST = '127.0.0.1'  # Server's IP address
    PORT = 12346  # Server's port number

    # Generate random data
    data = generate_random_data()
    for key, value in data.items():
        print(f"{key}: {value}")

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    # Send data to server
    send_data(client_socket, data)

    # Receive response from server
    deserialized_response = receive_data(client_socket)

    # Print each key-value pair
    for key, value in deserialized_response.items():
        print(f"{key}: {value}")

    # Close connection
    client_socket.close()


if __name__ == "__main__":
    main()
