import requests
from openai import OpenAI
from typing import List, Optional

from patientsim.utils import colorstr, log



class VLLMClient:
    def __init__(self, model: str, vllm_endpoint: str):
        # Initialize
        self.model = model
        self.vllm_endpoint = vllm_endpoint
        self._init_environment()
        self.histories = list()
        self.token_usages = dict()
        self.__first_turn = True
        self.__sanity_check()


    def _init_environment(self):
        """
        Initialize vLLM OpenAI-formatted client.
        """
        self.client = OpenAI(
            base_url=f"{self.vllm_endpoint}/v1",
            api_key='EMPTY'
        )


    def __sanity_check(self) -> None:
        response = requests.get(f'{self.vllm_endpoint}/v1/models')
        if response.status_code != 200:
            raise ValueError(colorstr("red", f"Failed to retrieve models: {response.text}"))
        
        models = response.json()
        if not models.get("data"):
            raise ValueError(colorstr("red", "No models found."))
        available_model_ids = [m['id'] for m in models['data']]
        if self.model not in available_model_ids:
            raise ValueError(colorstr("red", f"Model '{self.model}' not found in available models: {', '.join(available_model_ids)}"))


    def reset_history(self, verbose: bool = True) -> None:
        """
        Reset the conversation history.

        Args:
            verbose (bool): Whether to print verbose output. Defaults to True.
        """
        self.__first_turn = True
        self.histories = list()
        self.token_usages = dict()
        if verbose:
            log('Conversation history has been reset.', color=True)

    
    def __make_payload(self, user_prompt: str) -> List[dict]:
        """
        Create a payload for API calls to the model.

        Args:
            user_prompt (str): User prompt.

        Returns:
            List[dict]: Payload including prompts and image data.
        """
        payloads = list()
        user_contents = {"role": "user", "content": []}

        # User prompts
        user_contents["content"].append(
            {"type": "text", "text": user_prompt}
        )

        payloads.append(user_contents)
        
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

            if self.__first_turn:
                # System prompt
                if system_prompt:
                    self.histories.append({"role": "system", "content": system_prompt})
                
                # Greeting
                if greeting and self.__first_turn:
                    self.histories.append({"role": "assistant", "content": greeting})
                
                self.__first_turn = False

            # User prompt
            self.histories += self.__make_payload(user_prompt)
            
            # Model response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.histories,
                **kwargs
            )
            assistant_msg = response.choices[0].message
            self.histories.append({"role": assistant_msg.role, "content": assistant_msg.content})

            # Logging token usage
            if response.usage:
                self.token_usages.setdefault("prompt_tokens", []).append(response.usage.prompt_tokens)
                self.token_usages.setdefault("completion_tokens", []).append(response.usage.completion_tokens)
                self.token_usages.setdefault("total_tokens", []).append(response.usage.total_tokens)

            return assistant_msg.content
        
        except Exception as e:
            raise e
