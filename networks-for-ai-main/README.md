# Networks for AI

The emerging field of Networks for AI investigates design strategies for Next-G wireless communication networks specifically tailored to meet the large-scale AI workloads' unique communication needs. This paper introduces an innovative approach to partitioning the layers of large language models (LLMs) allowing the AI workload to be distributed across resource-constrained computing nodes within a cellular network. This split LLM architecture is implemented on an open-source E2E O-RAN testbed, which performs traffic classification and traffic steering to dynamically optimize network performance and address communication challenges posed by the distributed AI workload.

## Splitting the LLM

To partition the weights of the large language model (LLM), follow these steps:

1. **Use the Notebook:**
   Open and run the `PartitionGemma.ipynb` notebook. You may need to create a kaggle account and ask for permission in order to download the original weights of the models. Follow the instructions up to, but not including, the "Splitting Tests" section.

2. **Choose Your Model Variant:**
   By default, the notebook partitions the first layer from the rest of the model for the Gemma 2B variant. If you wish to work with the Gemma 7B variant:
   - In the second cell of the notebook, change the `VARIANT` variable to one of the 7B variants.
   - A list of available 7B Gemma variants can be found in the PyTorch section at [Kaggle: Gemma Models](https://www.kaggle.com/models/google/gemma).

3. **Locate the Partitions:**
   Once the model is partitioned, you should move the resultant model files to the desired location on your system where you plan to run them.

Follow the instructions outlined in the subsequent sections to utilize the partitioned model effectively.

## Recording Traffic
### Prerequisites
Before starting, update your settings as follows:
- In the `inference UE` script, adjust the `TXT_FILE` variable to the desired output file name.
- Set the `host` variable to the correct IP address.
- Ensure that the ports on the server and UE are aligned.

### Steps to Record Traffic
1. **Initialize the Server:**
   Run the following command to start the server:
   ```bash
   python inferenceServer.py
   ```

2. **Initialize the User Equipment (UE):**
   Start the UE using Streamlit:
   ```bash
   streamlit run inferenceUE.py
   ```

3. **Begin Recording:**
   Use `tcpdump` to start recording the traffic:
   ```bash
   sudo tcpdump -i utun6 -w recording_file.pcap host <host-ip> and tcp portrange 12346-65535
   ```

4. **Enter a Prompt:**
   Go to the Streamlit web app, and input a prompt of the desired length.

5. **End Recording:**
   Once the response appears on the screen, terminate the `tcpdump` recording and close the Streamlit web app using `Ctrl + C`.

   This action will generate a `.pcap` and a `.txt` file in the experiments folder with the specified names. To record extended conversations and traffic, simply issue multiple prompts before terminating the program.

## Analizing the results
[TO DO: add description of how to generate plots from pcap files]
