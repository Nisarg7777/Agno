import os
import speech_recognition as sr
import pyttsx3
from agno.agent import Agent
from agno.models.ollama import Ollama

# Initialize speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Function to capture voice input
def capture_voice_input():
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                input_text = recognizer.recognize_google(audio)
                print(f"You said: {input_text}")
                return input_text
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
                return ""
            except sr.RequestError:
                print("Could not request results from Google Speech Recognition service.")
                return ""
    except Exception as e:
        print(f"Error accessing microphone: {e}")
        return ""

# Function to output voice response
def speak_response(response):
    if response:
        print(f"Assistant says: {response}")
        tts_engine.say(response)
        tts_engine.runAndWait()

# Define the Voice AI Agent using Agno and Ollama Model
class VoiceAIgent:
    def __init__(self, model_id):
        # Initialize the Agno Agent with the Ollama model
        try:
            self.agent = Agent(
                name="Voice AI Agent",
                role="Assist the user with voice-based responses",
                model=Ollama(id=model_id),
                tools=[],  # Add any relevant tools if needed
                instructions="Assist with voice-based interaction, process commands and provide relevant answers",
                show_tool_calls=False,
                markdown=True
            )
        except Exception as e:
            print(f"Error initializing the agent: {e}")

    def process_input(self, input_text):
        try:
            # Process the input using the Agno Agent
            response = self.agent.run(input_text)
            
            # Check if response contains messages
            if hasattr(response, 'messages') and response.messages:
                # Extract the content from the assistant's message
                for message in response.messages:
                    if message.role == 'assistant':
                        return message.content.strip()  # Strip any extra spaces or newlines
                return "No response from assistant"
            else:
                return "No response"
        except Exception as e:
            print(f"Error processing input: {e}")
            return "Error processing input"

    def run(self):
        print("Voice AI Agent is running... Speak now!")
        while True:
            # Capture voice input
            input_text = capture_voice_input()
            if input_text.lower() == "exit":
                print("Exiting Voice AI Agent.")
                break

            # Process the input
            response = self.process_input(input_text)

            # Output the response as voice
            speak_response(response)

# Step 2: Run the agent
if __name__ == "__main__":
    # Update the model_id with the correct Ollama Mistral model ID
    model_id = "mistral"  # Use your local llama3.1 model ID
    voice_ai_agent = VoiceAIgent(model_id=model_id)
    voice_ai_agent.run()
