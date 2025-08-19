import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List, Optional

from patientsim.utils import log



class GeminiClient:
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self._init_environment(api_key)
        self.chat = None
        self._multi_turn_chat_already_set = False
        self.__first_turn = True


    def _init_environment(self, api_key: Optional[str] = None):
        """
        Initialize Goolge GCP Gemini client.

        Args:
            api_key (Optional[str]): API key for OpenAI. If not provided, it will
                                     be loaded from environment variables.
        """
        if not api_key:
            load_dotenv(override=True)
            api_key = os.environ.get("GOOGLE_API_KEY", None)
        self.client = genai.Client(api_key=api_key)


    def reset_history(self):
        """
        Reset the conversation history.
        """
        self.chat = None
        self._multi_turn_chat_already_set = False
        self.__first_turn = True
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
                 **kwargs) -> str:
        """
        Sends a chat completion request to the model with optional image input and system prompt.

        Args:
            user_prompt (str): The main user prompt or query to send to the model.
            system_prompt (Optional[str], optional): An optional system-level prompt to set context or behavior. Defaults to None.
            using_multi_turn (bool): Whether to structure it as multi-turn. Defaults to True.

        Raises:
            e: Any exception raised during the API call is re-raised.

        Returns:
            str: The model's response message.
        """
        try:
            if using_multi_turn:
                if self._multi_turn_chat_already_set and system_prompt:
                    if self.__first_turn:
                        log('Since the initial system prompt was already set, the current system prompt is ignored.', 'warning')
                        self.__first_turn = False
                    system_prompt = None

                if not self.chat:
                    self.chat = self.client.chats.create(
                        model=self.model,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt if system_prompt else None,
                            **kwargs
                        )
                    )
                    self._multi_turn_chat_already_set = True
                
                # User prompt and model response
                payloads = self.__make_payload(user_prompt)
                response = self.chat.send_message(payloads[0].parts)

            else:
                # To ensure empty history
                self.reset_history()

                # User prompt
                payloads = self.__make_payload(user_prompt)
                
                # System prompt and model response
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=payloads,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        **kwargs
                    )
                )

            return response.text
        
        except Exception as e:
            raise e