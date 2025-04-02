import gradio as gr
import requests
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the JSON file containing use cases
with open("voice_description_indian.json", "r") as file:
    usecases = json.load(file)

def get_api_token():
    username = os.getenv('USERNAME')
    password = os.getenv('PASSWORD')
    if not username or not password:
        raise ValueError("Environment variables USERNAME and PASSWORD must be set.")
    url = 'https://slabstech-dhwani-server.hf.space/v1/token'
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    payload = {"username": username, "password": password}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        logger.error(f"Failed to get API token: {response.text}")
        return None

def get_audio(input_text, usecase_id):
    try:
        api_token = get_api_token()
        if not api_token:
            return "Error: Failed to obtain API token."

        # Retrieve the selected use case
        usecase = next((uc for uc in usecases["usecases"] if uc["id"] == usecase_id), None)
        if not usecase:
            return f"Error: Use case with ID {usecase_id} not found."

        voice_description = usecase.get("voice_description")
        
        if not voice_description:
            return "Error: Missing voice description in the use case JSON."

        logger.info(f"Voice Description: {voice_description}")
        logger.info(f"Input Text: {input_text}")

        url = f"https://slabstech-dhwani-server.hf.space/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "accept": "application/json"
        }
        
        params = {
            "input": input_text,
            "voice": voice_description,
            "model": "ai4bharat/indic-parler-tts",
            "response_format": "mp3",
            "speed": 1.0
        }
        
        response = requests.post(url, headers=headers, params=params)
        
        if response.status_code == 200:
            logger.info(f"API request successful. Status code: {response.status_code}")
            audio_file_path = f"{usecase_id}_output.mp3"
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(response.content)
            logger.info(f"Audio file saved to: {audio_file_path}")
            return audio_file_path
        else:
            logger.error(f"API request failed. Status code: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception in get_audio: {e}")
        return None

demo = gr.Interface(
    fn=get_audio,
    inputs=[
        gr.Textbox(label="Enter Text", placeholder="Type your text here..."),
        gr.Dropdown(
            label="Select Use Case",
            choices=[uc["id"] for uc in usecases["usecases"]],
            type="value"
        )
    ],
    outputs=gr.Audio(label="Generated Audio"),
)

if __name__ == "__main__":
    try:
        demo.launch(show_error=True)
    except Exception as e:
        logger.error(f"Failed to launch Gradio demo: {e}")
