import numpy as np
import json
import socket
import base64
import torch


class NumPySerializer:
    @staticmethod
    def serialize(data):
        serialized_data = {}
        for key, value in data.items():
            # Convert bytes to Base64 encoded string
            serialized_data[key] = base64.b64encode(value.tobytes()).decode('utf-8')
        return json.dumps(serialized_data)

    @staticmethod
    def deserialize(data):
        deserialized_data = {}
        for key, value in json.loads(data).items():
            # Decode Base64 encoded string back to bytes
            deserialized_data[key] = np.frombuffer(base64.b64decode(value), dtype=np.int64)
        return deserialized_data


class TorchTensorSerializer:
    @staticmethod
    def serialize(data):
        serialized_data = {}
        for key, value in data.items():
            serialized_data[key] = {
                'type': str(value.dtype),
                'shape': value.shape,
                'data': base64.b64encode(value.cpu().detach().to(torch.float32).numpy().tobytes()).decode('utf-8')
            }
        return json.dumps(serialized_data)

    @staticmethod
    def deserialize(data):
        deserialized_data = {}
        for key, value in json.loads(data).items():
            tensor_shape = value['shape']
            tensor_bytes = base64.b64decode(value['data'])
            np_array = np.frombuffer(tensor_bytes, dtype=np.float32)  # Assuming float32 dtype for tensors
            deserialized_data[key] = torch.tensor(np_array).reshape(tensor_shape).to(eval(value['type']))
        return deserialized_data


def send_data(socket, data):
    if 'batch_size' in data.keys():
        serialized_data = json.dumps(data)
    else:
        serialized_data = TorchTensorSerializer.serialize(data)

    # Send serialized data to socket in one call
    socket.sendall(serialized_data.encode('utf-8'))

    # Notify receiver that data transmission is complete
    socket.sendall('END'.encode('utf-8'))

    # Wait for acknowledgment from the receiver
    try:
        ack = socket.recv(32).decode('utf-8')
        if ack != 'ACK':
            raise Exception("Did not receive acknowledgment from the receiver.")
    except socket.error as e:
        print(f"Socket error while waiting for ACK: {e}")


def receive_data(client_socket):
    buffer_size = 2048
    max_buffer_size = 65536
    min_buffer_size = 512
    data = ''
    while True:
        try:
            chunk = client_socket.recv(buffer_size).decode('utf-8')
            if not chunk:
                break  # Connection closed
            if chunk.endswith('END'):
                data += chunk[:-3]
                break
            data += chunk

            # Adjust buffer size based on the size of the last chunk received
            if len(chunk) < buffer_size and buffer_size > min_buffer_size:
                buffer_size = max(buffer_size // 2, min_buffer_size)
            elif len(chunk) == buffer_size and buffer_size < max_buffer_size:
                buffer_size = min(buffer_size * 2, max_buffer_size)
        except socket.timeout as e:
            print(f"Socket timeout: {e}")
            break
        except socket.error as e:
            print(f"Socket error: {e}")
            break

    try:
        # Send acknowledgment back to sender
        client_socket.sendall('ACK'.encode('utf-8'))
    except BrokenPipeError as e:
        print(f"BrokenPipeError while sending ACK: {e}")
    except socket.error as e:
        print(f"Socket error while sending ACK: {e}")

    # Chose one method below depending on data
    if 'batch_size' in json.loads(data).keys():
        received_data = json.loads(data)
    else:
        # Deserialize received Torch data
        received_data = TorchTensorSerializer.deserialize(data)
    return received_data

