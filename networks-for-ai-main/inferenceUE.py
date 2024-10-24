import streamlit as st
import os
import kagglehub
import torch
import argparse
from tools.config import get_config_for_7b, get_config_for_2b
from tools.model import Sampler, Embedding, precompute_freqs_cis
from tools.tokenizer import Tokenizer
from tools.model_utils import GemmaLayerModel, GemmaLastLayerModel, load_model, generation, dummy_generation

def parse_args():
    parser = argparse.ArgumentParser(description="Introduce host and port of starting remote node.")
    parser.add_argument("--host_next", type=str, default='127.0.0.1', help="Host IP from next layer")
    parser.add_argument("--port_next", type=int, default=12346, help="Port from next layer")
    parser.add_argument("--add_lastlayer", action="store_true", default=False, help="Add last decoder layer to the UE side")

def initialize_last_layer(config):
    last_layer_model = GemmaLastLayerModel(config) 
    load_model(last_layer_model, f'./weights/{VARIANT}/layer_model_{config.num_hidden_layers - 1}.pth')
    last_layer_model.to(device)
    last_layer_model.eval()
    return last_layer_model
    

if __name__ == "__main__":
    args = parse_args()
    next_layer = {'host': args.host_next, 'port': args.port_next}
    include_ll = args.add_lastlayer # Include last decoder layer

    # Choose variant and machine type
    VARIANT = '2b-it'
    MACHINE_TYPE = 'cuda'
    TXT_FILE = 'rec8.txt'
    # Define host and port
    host = '10.75.11.68'
    port = 12346

    # Download tokenizer
    home_dir = os.path.expanduser("~")
    weights_dir = os.path.join(home_dir, ".cache", "kagglehub", "models", "google", "gemma", "pyTorch", VARIANT)
    weights_dir = os.path.join(weights_dir, "1") if '1.1' in VARIANT else os.path.join(weights_dir, "2")
    if not os.path.exists(weights_dir):
        kagglehub.login() # API KEY: 5cb66339276d4bea7ba59ca714d28f6b
        weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')
    tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
    assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

    # Define config
    config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    config.tokenizer = tokenizer_path
    config.quant = 'quant' in VARIANT
    torch.set_default_dtype(config.get_dtype())
    device = torch.device('cuda' if torch.cuda.is_available() and MACHINE_TYPE == 'cuda' else 'cpu')

    # Initialize models
    first_layer_model = GemmaLayerModel(config) 

    # Load trained weights into models
    load_model(first_layer_model, f'./weights/{VARIANT}/layer_model_0.pth')
        
    # Move models to the appropriate device (e.g., GPU or CPU)
    first_layer_model.to(device)
        
    # Ensure models are in evaluation mode
    first_layer_model.eval()

    # Hidden size needs to be divisible by the number of heads
    assert config.hidden_size % config.num_attention_heads == 0

    # Prepare inference auxiliary structures
    max_seq_len = config.max_position_embeddings
    head_dim = config.head_dim
    vocab_size = config.vocab_size
    tokenizer = Tokenizer(config.tokenizer)
    embedder = Embedding(vocab_size, config.hidden_size, config.quant)
    load_model(embedder, f'./weights/{VARIANT}/embedding_weights.pth')
    embedder.to(device)
    embedder.eval()
    model = [first_layer_model]
    if include_ll:
        last_layer_model = initialize_last_layer(config)
        model.append(last_layer_model)
    # Initialize sampler
    sampler = Sampler(vocab_size).to(device)
    # Pre-compute rotary embedding table.
    rope_theta = getattr(config, 'rope_theta', 10000)
    prec_freqs_cis = precompute_freqs_cis(head_dim,
                                    max_seq_len * 2,
                                    theta=rope_theta).to(device)

    # Chat templates
    USER_CHAT_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n'
    MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n{prompt}<end_of_turn>\n'

    # Streamlit app setup
    st.title("Gemma Model Text Generation")

    # Input textbox for user prompts
    user_input = st.text_area("Enter your prompt", "What is the best city in Europe?")

    # Chat history (store this in the session state to persist across reruns)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ""

    # Display the chat history
    #st.text_area("Chat History", st.session_state.chat_history, height=250)

    # Button to generate response
    if st.button("Generate Response"):
        # Call the generation function
        # Append new user input to chat history
        st.session_state.chat_history += USER_CHAT_TEMPLATE.format(prompt=user_input + '<start_of_turn>model\n')
        model_response = generation(
            host,
            port,
            tokenizer,
            config,
            model,
            embedder,
            sampler,
            prec_freqs_cis,
            prompts=st.session_state.chat_history,
            device=device,
            output_len=200  # Adjust as needed
        )

        # Record context
        with open('./experiments/' + TXT_FILE, 'w') as file:
            # Write model interaction to file
            file.write(st.session_state.chat_history + model_response)

        # Append model response to chat history
        st.session_state.chat_history += MODEL_CHAT_TEMPLATE.format(prompt=model_response)
        

        
        # Update the chat history display
        st.text_area("Chat History", st.session_state.chat_history, height=250)


