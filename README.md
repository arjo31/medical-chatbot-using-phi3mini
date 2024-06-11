# Medical Chatbot using Microsoft's Phi3 Mini model

## Description : 
An end to end Medical Chatbot Using Microsoft's Phi3 Mini model downloaded from Ollama. The chatbot uses RAG and RetreivalQA to answer queries based on the 5 volumes of Gale's Encyclopedia of medicines. We also use the All MiniLM L6 model as the word embedding layer from HuggingFace. Currently, the app can be run locally on your host computer. Looking for ways to deploy it.

## Tech Stack : 
1. Python
2. Streamlit
3. Langchain
4. Microsoft Phi3 Mini LLM Model
5. Ollama
6. FAISS Vector Database
7. All MiniLM L6 model as a sentence transformer.
8. HuggingFace

## Steps to run : 
1. For Windows : Download Ollama from this [link](https://ollama.com/).
   
   For Linux : Run this command in the terminal :
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
3. Run this command to download the Microsoft Phi3:mini model
   ```bash
   ollama run phi3:mini
   ```
   You can try out other models supported by Ollama from this [link](https://ollama.com/library).
4. Create a conda virtual environment using the following command :
   ```bash
   conda create -p <ENV_NAME> python==3.12 -y
   ```
5. Activate the conda environment using this command :
   ```bash
   conda activate <PATH_TO_VIRTUAL_ENVIRONMENT>
   ```

   To deactivate the environment use :
   ```bash
   conda deactivate
   ```
6. Install all the required libraries from requirements.txt
   ```bash
   pip install -r requirements.txt
   ```
7. Create a Langsmith API Key from this [link][https://smith.langchain.com] and get the Langsmith Project Name after creating a new project, to track how the responses are being managed and the time taken by the model to give the response. Store all important files in a .env file.
8. Run the following command to run the Streamlit App
   ```bash
   streamlit run app.py
   ```

   The app will now run in your local machine. Please feel free to test it out and suggest any changes.
