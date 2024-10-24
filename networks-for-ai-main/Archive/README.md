# Network Utils

The `network_utils` module provides classes and functions for handling network communication, message serialization, and message parsing.

## NetworkConnection

The `NetworkConnection` class represents a network connection to a remote host. It provides methods for establishing a connection, sending data, and receiving data over the network.

### Methods

- `connect()`: Establishes a connection to the remote host specified by the host address and port.
- `send_data(message)`: Sends a message over the network to the remote host.
- `receive_data()`: Receives a message from the remote host over the network.
- `close()`: Closes the network connection.

## Message

The `Message` class represents a message that can be sent over the network. It encapsulates the message type and data, and provides methods for serializing and deserializing messages. The maximum message length is 4.29 GB.

### Methods

- `to_json()`: Serializes the message to JSON format.
- `from_json(json_string)`: Deserializes a JSON string into a Message object.
- `from_raw_message(raw_message)`: Parses a raw message (bytes) into a Message object.

## MessageType

The `MessageType` class defines constants for different types of messages. These constants are used to identify the type of message being sent or received. Message types include:

- `hidden_state`
- `CIS`
- `KV_index`
- `KV_cache`
- `mask`

## Functionality

- Establishing network connections between client and server applications.
- Sending and receiving messages over the network.
- Serializing messages to JSON format for transmission.
- Deserializing messages from JSON format into Message objects.
- Parsing raw message bytes into Message objects.

## Usage

### Client

1. Create a `NetworkConnection` instance with the host address and port of the server.
2. Connect to the server using the `connect()` method.
3. Generate or obtain data to send.
4. Create a `Message` object with the message type and data.
5. Send the message using the `send_data()` method.
6. Receive response messages using the `receive_data()` method.

### Server

1. Create a server socket using the `create_server()` function.
2. Accept incoming connections using the `accept_connection()` function.
3. Handle client connections in separate processes or threads.
4. Receive messages from clients using the `receive_data()` method.
5. Process received messages and send response messages.

# Example Server & Client

### `client.py`

This script acts as a client and connects to a server to send and receive messages of different types.

#### Usage: 
To use the client script:

1. Ensure that the `network_utils.py` module is available in the same directory or in the Python path. This module provides utility functions for managing network connections and message serialization.

2. Update the `host` and `port` variables in the script to specify the IP address and port number of the server.

3. Customize the message generation and processing logic as needed in the `__main__` block. This block defines how messages are created, sent to the server, and responses are handled.

4. Run the `client.py` script using Python 3.x. It will connect to the specified server, send messages, and display the responses received from the server.

### `server.py`

The `server.py` script contains the server-side logic for handling client connections, receiving messages, and sending responses. This script uses mutli-processing to allow multiple connections from servers.

#### Functions:

- `handle_client`: Function responsible for handling client connections. It receives messages from clients, processes them, and sends responses.

- `start_server`: Function to start the server. It creates a server socket, listens for incoming connections, and spawns a new process to handle each client connection.

#### Usage

To use the server script:

1. Ensure that the `network_utils.py` module is available in the same directory or in the Python path. This module provides utility functions for managing network connections and message serialization.

2. Update the `host` and `port` variables in the script to specify the desired host IP address and port number for the server to listen on.

3. Customize the message handling logic in the `handle_client` function according to the application requirements. This function defines how received messages are processed and responses are generated.

4. Run the `server.py` script using Python 3.x. The server will start listening for incoming connections on the specified host and port.

# Dependencies

- Python 3.x
