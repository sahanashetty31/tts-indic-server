import gradio as gr
import requests
import logging
import json
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the JSON file containing use cases
with open("voice_description_indian.json", "r") as file:
    usecases = json.load(file)
    logger.debug(f"Loaded {len(usecases['usecases'])} use cases from JSON file")

# Function to obtain an access token
def get_access_token(username, password):
    try:
        url_token = "https://slabstech-dhwani-server.hf.space/v1/token"
        payload = {"username": username, "password": password}
        
        response_token = requests.post(url_token, json=payload)
        response_token.raise_for_status()
        token_data = response_token.json()
        if 'token' in token_data:
            token = token_data['token']
        elif 'access_token' in token_data:
            token = token_data['access_token']
        else:
            logger.error("Token not found in response. Full response: %s", response_token.text)
            return None
            
        logger.debug("Successfully acquired access token")
        return token

    except requests.exceptions.HTTPError as e:
        logger.error(f"Token request HTTP error: {e.response.status_code} - {e.response.text}")
        return None

# Function to send text input to the API and retrieve the audio file
def get_audio(input_text, usecase_id, token):
    try:
        logger.info(f"Starting audio generation request")
        logger.debug(f"Inputs - Text: '{input_text}', UseCase ID: {usecase_id}")

        # Use case validation
        logger.debug(f"Looking up use case ID: {usecase_id}")
        usecase = next((uc for uc in usecases["usecases"] if str(uc["id"]) == str(usecase_id)), None)
        if not usecase:
            logger.error(f"Use case {usecase_id} not found in available options")
            return f"Error: Invalid use case ID {usecase_id}"
            
        logger.debug(f"Found use case: {usecase['voice_description']}")

        # API request
        url_audio = "https://slabstech-dhwani-server.hf.space/v1/audio/speech"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        # Construct URL with query parameters
        query_params = {
            "input": input_text,
            "voice": usecase["voice_description"],
            "model": "ai4bharat/indic-parler-tts",
            "response_format": "mp3",
            "speed": 1
        }
        
        url_audio_with_params = f"{url_audio}?input={input_text}&voice={usecase['voice_description']}&model=ai4bharat%2Findic-parler-tts&response_format=mp3&speed=1"
        
        logger.debug(f"Sending request to {url_audio_with_params}")
        
        response_audio = requests.post(url_audio_with_params, headers=headers, stream=True)
        
        try:
            response_audio.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Audio API HTTP error: {e.response.status_code} - {e.response.text}")
            return f"API Error: {e.response.status_code} - {e.response.text}"

        # File handling
        audio_file_path = f"usecase_{usecase['id']}_output.mp3"  # Short and meaningful filename
        logger.debug(f"Saving audio to: {audio_file_path}")
        
        try:
            with open(audio_file_path, "wb") as audio_file:
                for chunk in response_audio.iter_content(chunk_size=1024):
                    if chunk:
                        audio_file.write(chunk)
            logger.info(f"Successfully saved audio file: {audio_file_path}")
            return audio_file_path
        except IOError as e:
            logger.error(f"File save error: {str(e)}")
            return f"Error saving file: {str(e)}"

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return f"Critical error: {str(e)}"

# Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        username_input = gr.Textbox(label="Username")
        password_input = gr.Textbox(label="Password", type="password")
    
    input_text = gr.Textbox(label="Input Text")
    usecase_dropdown = gr.Dropdown(
        label="Use Case",
        choices=[f"{uc['id']}: {uc['voice_description']}" for uc in usecases["usecases"]],
    )
    
    generate_button = gr.Button("Generate Audio")
    audio_output = gr.Audio(label="Output")

    def process_request(input_text, usecase_entry, username, password):
        logger.debug(f"Processing request - User: {username}, Selection: {usecase_entry}")
        
        # Extract use case ID from the selected entry
        usecase_id = usecase_entry.split(":")[0].strip()
        
        # Obtain access token
        token = get_access_token(username, password)
        if not token:
            return "Error: Unable to authenticate"
        
        return get_audio(input_text, usecase_id, token)

    generate_button.click(
        fn=process_request,
        inputs=[input_text, usecase_dropdown, username_input, password_input],
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch(share=True)
