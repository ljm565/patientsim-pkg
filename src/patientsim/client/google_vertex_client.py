import os
import time
from dotenv import load_dotenv
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

from patientsim.utils import log
from patientsim.utils.common_utils import exponential_backoff



class GeminiVertexClient:
    def __init__(self,
                 model: str,
                 project_id: Optional[str] = None,
                 project_location: Optional[str] = None,
                 vertex_credentials: Optional[str] = None):
        # Initialize
        self.model = model
        self._init_environment(project_id, project_location, vertex_credentials)
        self.histories = list()
        self.__first_turn = True


    def _init_environment(self,
                          project_id: Optional[str] = None,
                          project_location: Optional[str] = None,
                          vertex_credentials: Optional[str] = None) -> None:
        """
        Initialize Goolge GCP Gemini client.

        Args:
            project_id (Optional[str]): Google Cloud project ID. If not provided, it will
                                        be loaded from environment variables.
            project_location (Optional[str]): Google Cloud project location. If not provided,
                                              it will be loaded from environment variables.
            vertex_credentials (Optional[str]): Path to the Google Cloud credentials file. If not provided,
                                               it will be loaded from environment variables.
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None) if vertex_credentials is None else vertex_credentials
        if not project_id or not project_location:
            load_dotenv(override=True)
            project_id = os.environ.get("GOOGLE_PROJECT_ID", None) if not project_id else project_id
            project_location = (
                os.environ.get("GOOGLE_PROJECT_LOCATION", None)
                if not project_location
                else project_location
            )
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=project_location,
            http_options=HttpOptions(api_version="v1"),
        )


    def reset_history(self, verbose: bool = True) -> None:
        """
        Reset the conversation history.

        Args:
            verbose (bool): Whether to print verbose output. Defaults to True.
        """
        self.__first_turn = True
        self.histories = list()
        if verbose:
            log('Conversation history has been reset.', color=True)


    def __make_payload(self, user_prompt: str) -> List[types.Content]:
        """
        Create a payload for API calls to the Gemini model.

        Args:
            user_prompt (str): User prompt.

        Returns:
            List[types.Content]: Payload including prompts and image data.
        """
        payloads = list()    
        
        # User prompts
        user_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=user_prompt)]
        )
        
        payloads.append(user_content)
        
        return payloads


    def __call__(self,
                 user_prompt: str,
                 system_prompt: Optional[str] = None,
                 using_multi_turn: bool = True,
                 greeting: Optional[str] = None,
                 verbose: bool = True,
                 **kwargs) -> str:
        """
        Sends a chat completion request to the model with optional image input and system prompt.

        Args:
            user_prompt (str): The main user prompt or query to send to the model.
            system_prompt (Optional[str], optional): An optional system-level prompt to set context or behavior. Defaults to None.
            using_multi_turn (bool): Whether to structure it as multi-turn. Defaults to True.
            greeting (Optional[str]): An optional greeting message to include in the conversation. Defaults to None.
            verbose (bool): Whether to print verbose output. Defaults to True.

        Raises:
            e: Any exception raised during the API call is re-raised.

        Returns:
            str: The model's response message.
        """
        try:
            # To ensure empty history
            if not using_multi_turn:
                self.reset_history(verbose)

            # Greeting
            if greeting and self.__first_turn:
                self.histories.append(types.Content(role='model', parts=[types.Part.from_text(text=greeting)]))
                self.__first_turn = False

            # User prompt
            self.histories += self.__make_payload(user_prompt)

            # System prompt and model response, including handling None cases
            count = 0
            max_retry = kwargs.get('max_retry', 5)
            while 1:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.histories,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        **kwargs
                    )
                )

                # After the maximum retries
                if count >= max_retry:
                    replace_text = 'Could you tell me again?'
                    self.histories.append(types.Content(role='model', parts=[types.Part.from_text(text=replace_text)]))
                    return replace_text

                if response.text == None:
                    wait_time = exponential_backoff(count)
                    time.sleep(wait_time)
                    count += 1
                    continue
                else:
                    break
            
            self.histories.append(types.Content(role='model', parts=[types.Part.from_text(text=response.text)]))
            return response.text
        
        except Exception as e:
            raise e
