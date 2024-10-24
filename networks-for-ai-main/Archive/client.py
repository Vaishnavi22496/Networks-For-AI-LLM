import network_utils
import random
import string

host = '127.0.0.1'
port = 12345


def generate_random_data(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


if __name__ == "__main__":
    # Iterate over attributes of MessageType class
    for message_type_name in dir(network_utils.MessageType):
        message_type = getattr(network_utils.MessageType, message_type_name)
        if isinstance(message_type, str):
            # Create a new NetworkConnection instance for each message type
            client_connection = network_utils.NetworkConnection(host, port)

            try:
                # Connect to the server
                client_connection.connect()

                # Generate random data
                data_length = random.randint(10**6, 10**7)  # Random data length between 10 and 100
                random_data = generate_random_data(data_length)
                """
                print("Sending message type:", message_type,
                      "with length:", data_length,
                      "and contents:", random_data)
                """
                # Create the message
                '''
                # Transfer tensor from GPU to CPU
                random_tensor_cpu = random_tensor.cpu()

                # Convert tensor to numpy array or bytes for serialization
                serialized_data = random_tensor_cpu.numpy().tobytes()
                message = network_utils.Message(message_type, serialized_data)
                '''
                message = network_utils.Message(message_type, random_data)
                print("Sending message type:", message.message_type)
                print("Message length:", data_length)
                # print("Sending contents:", message.data)

                # Send the message
                client_connection.send_data(message)

                # Receive response
                message_type, data_length, received_data = client_connection.receive_data()
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
                    else:
                        print("Failed to parse message.")
                else:
                    print("Invalid message received:", received_data)

            finally:
                # Close the connection when done
                client_connection.close()


