import socket
import json
import struct


class NetworkConnection:
    HEADER_SIZE = 20  # Size of the header in bytes

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.socket.connect((self.host, self.port))

    def send_data(self, message):
        serialized_message = message.to_json().encode()
        message_type = message.message_type.encode().ljust(Message.MESSAGE_TYPE_SIZE,
                                                           b'\x00')  # Pad message type to fixed size
        data_length = len(serialized_message)
        header = struct.pack("!16sI", message_type,
                             data_length)  # Pack message type as string and data length as 4-byte unsigned integer
        # print(f"sending header: {header}")
        self.socket.sendall(header + serialized_message)

    def receive_data(self):
        try:
            header = self.socket.recv(self.HEADER_SIZE)
            if not header:
                return None
            # print("Received header:", header)
            # Unpack message type and data length from the header
            message_type, data_length = struct.unpack("!16sI", header)
            # print(f'Message Type: {message_type}, data length: {data_length}')
            received_data = b''
            while len(received_data) < data_length:
                remaining_bytes = data_length - len(received_data)
                chunk = self.socket.recv(min(remaining_bytes, 1024))
                if not chunk:
                    break
                received_data += chunk

            # Decode the received data as JSON
            return message_type.decode().strip(), data_length, received_data.decode()
        except (socket.error, struct.error) as e:
            # print("Error receiving data:", e)
            return None, None, None

    def close(self):
        self.socket.close()


def create_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    return server_socket


def accept_connection(server_socket):
    client_socket, client_address = server_socket.accept()
    network_connection = NetworkConnection("", 0)  # Create a dummy NetworkConnection instance
    network_connection.socket = client_socket  # Assign the client socket to the NetworkConnection
    return network_connection, client_address


class MessageType:
    hidden_state = "hidden_state"
    CIS = "CIS"
    KV_index = "KV_index"
    KV_cache = "KV_cache"
    mask = "mask"


class Message:
    MESSAGE_TYPE_SIZE = 16  # Size of the message type field in bytes
    DATA_LENGTH_SIZE = 4     # Size of the data length field in bytes

    def __init__(self, message_type, data):
        self.message_type = message_type.ljust(self.MESSAGE_TYPE_SIZE)[:self.MESSAGE_TYPE_SIZE]  # Ensure fixed size
        self.data = data
        self.data_length = len(data)  # Store the length of the data

    def to_json(self):
        data_length = len(self.data)
        # Pack message type as string and data length as 4-byte unsigned integer
        header = struct.pack("!16sI", self.message_type.encode(), data_length)
        return json.dumps({"header": header.hex(), "data": self.data})

    @staticmethod
    def from_json(json_string):
        try:
            parsed_data = json.loads(json_string)
            header = bytes.fromhex(parsed_data["header"])
            message_type = header[:Message.MESSAGE_TYPE_SIZE].decode().strip()
            data_length = struct.unpack("!I", header[Message.MESSAGE_TYPE_SIZE:])[0]
            return Message(message_type, parsed_data["data"][:data_length]), parsed_data["data"][data_length:]
        except (json.JSONDecodeError, KeyError, IndexError, struct.error) as e:
            print("Error parsing JSON:", e)
            return None, None

    @staticmethod
    def from_raw_message(raw_message):
        try:
            header = raw_message[:Message.MESSAGE_TYPE_SIZE + Message.DATA_LENGTH_SIZE]
            message_type = header[:Message.MESSAGE_TYPE_SIZE].decode().strip()
            data_length = struct.unpack("!I", header[Message.MESSAGE_TYPE_SIZE:])[0]
            return raw_message[Message.MESSAGE_TYPE_SIZE +
                               Message.DATA_LENGTH_SIZE:Message.MESSAGE_TYPE_SIZE +
                                                        Message.DATA_LENGTH_SIZE + data_length], \
                   raw_message[Message.MESSAGE_TYPE_SIZE + Message.DATA_LENGTH_SIZE + data_length:]
        except (IndexError, struct.error) as e:
            print("Error parsing raw message:", e)
            return None, None

