import network_utils
import multiprocessing
import random
import string


host = '127.0.0.1'
port = 12345


def generate_random_data(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def handle_client(client_socket, client_address):
    print("Connection established with", client_address)

    try:
        # Receive message from client
        message_type, data_length, received_data = client_socket.receive_data()
        if received_data:
            print("Received message type:", message_type)
            '''
            # Deserialize received data into tensor
            received_tensor = torch.from_numpy(np.frombuffer(received_data, dtype=np.float32)).reshape(
                tensor_size)
            '''
            # Parse received data using Message.from_json to extract the actual message content
            message, _ = network_utils.Message.from_json(received_data)
            if message:
                message_content = message.data
                # print("Received message contents:", message_content)
                ascii_char_count = sum(1 for char in message_content if ord(char) < 128)
                print("Number of ASCII characters:", ascii_char_count)

                # Generate response with the same message type but different length and data
                data_length = random.randint(10**3, 10**4)  # Random data length between 10 and 100
                random_data = generate_random_data(data_length)
                response_message = network_utils.Message(message_type, random_data)

                # Send response to client
                '''
                # Transfer tensor from GPU to CPU
                random_tensor_cpu = random_tensor.cpu()

                # Convert tensor to numpy array or bytes for serialization
                serialized_data = random_tensor_cpu.numpy().tobytes()
                response_message = network_utils.Message(message_type, serialized_data)
                '''
                print("Sending message type:", message_type)
                print("Message length:", data_length)
                # print("Sending message contents:", random_data)
                client_socket.send_data(response_message)
            else:
                print("Failed to parse received message.")
        else:
            print("Invalid message received:", received_data)

    finally:
        client_socket.close()


def start_server():
    server_socket = network_utils.create_server(host, port)
    print("Server listening on", (host, port))

    while True:
        client_socket, client_address = network_utils.accept_connection(server_socket)

        # Create a new process for each incoming connection
        client_process = multiprocessing.Process(target=handle_client, args=(client_socket, client_address))
        client_process.start()


if __name__ == "__main__":
    start_server()
