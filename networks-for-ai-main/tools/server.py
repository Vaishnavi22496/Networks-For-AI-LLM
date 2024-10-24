import socket
import json
import numpy as np
from network_utils import send_data, receive_data
import signal
import sys
import torch


def handle_client(client_socket):
    # Receive data from client
    received_data = receive_data(client_socket)

    # Print each key-value pair
    for key, value in received_data.items():
        print(f"{key}: {value}")

    # Generate random hidden_state of size 100
    hidden_state = torch.rand(1, 7, 10)

    # Prepare response
    response = {'hidden_state': hidden_state}
    # print('Response message: ', response)

    # Send response back to client using existing connection
    send_data(client_socket, response)

    # Close client connection
    client_socket.close()


def start_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}...")

    def signal_handler(sig, frame):
        print('Closing server...')
        server_socket.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            client_socket, client_addr = server_socket.accept()
            print(f"Accepted connection from {client_addr[0]}:{client_addr[1]}")
            handle_client(client_socket)
    finally:
        server_socket.close()


if __name__ == "__main__":
    HOST = '127.0.0.1'  # Localhost
    PORT = 12346  # Port number
    start_server(HOST, PORT)
