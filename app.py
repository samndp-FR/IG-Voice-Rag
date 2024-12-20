from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import logging
from src.core.chat.service import ChatService
from src.core.voice.service import VoiceService

class WebApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, 
                               cors_allowed_origins="*",
                               logger=True, 
                               engineio_logger=True)
        self.chat_service = ChatService()
        self.voice_service = VoiceService()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.socketio.on('message')
        def handle_message(data):
            try:
                if data['type'] == 'text':
                    # Handle text message
                    response = self.chat_service.process_chat(data['message'])
                    emit('response', {
                        'response': response.response,
                        'sources': response.sources  # Include video URLs
                    })

                elif data['type'] == 'voice':
                    # Handle voice message
                    voice_data = data['message'].split(',')[1]
                    audio_bytes = base64.b64decode(voice_data)
                    
                    # Convert voice to text
                    text_result = self.voice_service.speech_to_text(audio_bytes)
                    if not text_result.success:
                        raise Exception("Failed to convert speech to text")
                        
                    # Get chat response with video references
                    chat_response = self.chat_service.process_chat(text_result.content)
                    
                    # Convert response to voice
                    voice_response = self.voice_service.text_to_speech(chat_response.response)
                    if not voice_response.success:
                        raise Exception("Failed to convert text to speech")
                        
                    # Send audio response with sources
                    audio_base64 = base64.b64encode(voice_response.content).decode('utf-8')
                    emit('response', {
                        'response': f"data:audio/mpeg;base64,{audio_base64}",
                        'text': chat_response.response,
                        'sources': chat_response.sources  # Include video URLs
                    })

            except Exception as e:
                self.logger.error(f"Error handling message: {str(e)}")
                emit('response', {'response': "Sorry, something went wrong. Please try again."})

    def run(self, port=4000, debug=True):
        self.socketio.run(self.app, port=port, debug=debug)

if __name__ == '__main__':
    app = WebApp()
    app.run()