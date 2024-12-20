from dataclasses import dataclass
from typing import List, Dict
from langchain_openai import AzureOpenAIEmbeddings
from supabase import create_client
import logging
import sys
from src.config.settings import get_settings

@dataclass
class SearchResult:
    content: str
    url: str
    similarity: float

class SimilaritySearch:
    def __init__(self):
        self.settings = get_settings()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add handler if not already added
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.supabase = create_client(
            self.settings.supabase_url,
            self.settings.supabase_key
        )
        
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.settings.ada_endpoint,
            api_key=self.settings.ada_api_key,
            api_version=self.settings.azure_api_version,
            model=self.settings.ada_deployment_name
        )
        
        self.logger.info("Search service initialized")

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        try:
            self.logger.info(f"Performing search for query: {query}")
            
            # Generate embedding for the query
            embedding = self.embeddings.embed_query(query)
            
            # Perform similarity search via Supabase RPC
            response = self.supabase.rpc(
                'match_video_content',
                {
                    'query_embedding': embedding,
                    'match_threshold': 0.8,  # Slightly lowered to ensure some results
                    'match_count': limit
                }
            ).execute()

            # Log search results
            self.logger.info(f"Found {len(response.data) if response.data else 0} results")

            # Handle no results
            if not response.data:
                self.logger.warning("No matches found for the query")
                return []

            # Process and sort results
            results = []
            for item in response.data:
                # Ensure we have all necessary fields
                result = {
                    'content': item.get('content', ''),
                    'url': item.get('url', ''),
                    'similarity': item.get('similarity', 0.0)
                }
                
                # Log individual match details
                self.logger.info(
                    f"Match found - "
                    f"Similarity: {result['similarity']:.3f}, "
                    f"URL: {result['url']}"
                )
                
                results.append(result)

            # Sort results by similarity in descending order
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results

        except Exception as e:
            # Ensure logging works even if something goes wrong
            print(f"Error in search: {str(e)}")
            if self.logger:
                self.logger.error(f"Search error: {str(e)}", exc_info=True)
            return []