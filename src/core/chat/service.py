# chat_service.py

from dataclasses import dataclass, field
from typing import Optional, List
import logging
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.search.service import SimilaritySearch
from src.config.settings import get_settings

@dataclass
class ChatResponse:
    response: str
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None
    url_message: str = ""

    def to_dict(self):
        """Convert the response to a dictionary format"""
        return {
            "response": self.response,
            "url": self.sources[0] if self.sources else "",
            "url_message": self.url_message,
            "error": self.error
        }

class ChatService:
    def __init__(self):
        self.settings = get_settings()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.chat_model = AzureChatOpenAI(
            azure_endpoint=self.settings.gpt4_endpoint,
            api_key=self.settings.gpt4_api_key,
            api_version=self.settings.azure_api_version,
            model=self.settings.gpt4_deployment_name,
            temperature=0.7
        )
        self.search = SimilaritySearch()

    def process_chat(self, query: str) -> ChatResponse:  # Fixed indentation here
        try:
            self.logger.info(f"Processing query: {query}")

            # Perform the search query
            search_results = self.search.search(query)
            self.logger.info(f"Found {len(search_results)} search results")

            # Check if there are search results
            if not search_results:
                return ChatResponse(
                    response="Sorry love, I haven't made a video about that yet",
                    sources=['']
                )

            # Get top match
            result = search_results[0]
            script_content = result.get('content', 'No content available').strip()
            top_url = result.get('url', '#')

            # Format context to emphasize script content
            context_message = (
                f"VIDEO SCRIPT CONTENT:\n{script_content}\n\n"
                f"IMPORTANT: Read the above script carefully. Only use tips and advice that appear in this script."
                f"\n\nQuestion from user: {query}"
            )

            # System message
            system_message = (
                "You are Chelsea, a quirky goat who's literally the GOAT of babysitting! "
                "IMPORTANT - Follow these steps exactly:\n"
                "1) READ the entire video script above carefully\n"
                "2) Structure your response  like this:\n"
                "   - Quick acknowledgement of the problem\n"
                "   - Say 'I literally made a video about this!'\n"
                "   - Don't say anything else than that, don't try to provide an answer'\n"
                "   - End with 'okay?' or 'trust me on this one!'\n"
                "3) CRITICAL RULES:\n"
                "   - Only use advice that appears word-for-word in the script\n"
                "   - Be playful, quirky and confident (use 'okay?', 'literally')\n"
                "   - Keep it super brief and snappy\n"
                "   - Never add advice that isn't in the script"
            )

            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=context_message)
            ]

            # Generate response using Azure Chat
            response = self.chat_model.invoke(messages)

            # Create URL message
            url_message = f"Watch the full video here: {top_url}" if top_url and top_url != '#' else ""

            # Return ChatResponse with separate response and URL
            return ChatResponse(
                response=response.content,
                sources=[top_url],
                url_message=url_message
            )

        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}", exc_info=True)
            return ChatResponse(
                response="Oh my goat! Something went wrong! Can you try asking that again?",
                sources=[],
                error=str(e)
            )