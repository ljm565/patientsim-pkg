import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List, Optional

from patientsim.utils import log


class GPTAzureClient:
    def __init__(self, model: str, api_key: Optional[str] = None, azure_endpoint: Optional[str] = None):
        self.model = model
        self._init_environment(api_key, azure_endpoint)
        self.histories = list()
        self._multi_turn_system_prompt_already_set = False
        self.__first_turn = True

    def _init_environment(self, api_key: Optional[str] = None, azure_endpoint: Optional[str] = None):
        """
        Initialize OpenAI client.

        Args:
            api_key (Optional[str]): API key for OpenAI. If not provided, it will
                                     be loaded from environment variables.
            azure_endpoint (Optional[str]): Azure endpoint for OpenAI. If not provided,
                                           it will be set to a default value.
        """
        if not api_key:
            load_dotenv(override=True)
            api_key = os.environ.get("OPENAI_API_KEY", None)

        if not azure_endpoint:
            load_dotenv(override=True)
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)

        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-10-21",
        )

    def reset_history(self, keep_system_prompt: bool = False):
        """
        Reset the conversation history.

        Args:
            keep_system_prompt (bool): Whether to retain the system prompt after reset.
                                       Defaults to False.
        """
        self.__first_turn = True
        system_message = None
        if keep_system_prompt:
            for msg in self.histories:
                if msg["role"] == "system":
                    system_message = msg
                    break
        else:
            self._multi_turn_system_prompt_already_set = False

        self.histories = []
        if system_message:
            self.histories.append(system_message)
            log("Conversation history has been reset except for the system prompt.", color=True)
        else:
            log("Conversation history has been reset.", color=True)

    def __make_payload(self, user_prompt: str) -> List[dict]:
        """
        Create a payload for API calls to the GPT model.

        Args:
            user_prompt (str): User prompt.

        Returns:
            List[dict]: Payload including prompts and image data.
        """
        payloads = list()
        user_contents = {"role": "user", "content": []}

        # User prompts
        user_contents["content"].append({"type": "text", "text": user_prompt})

        payloads.append(user_contents)

        return payloads

    def __call__(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        using_multi_turn: bool = True,
        **kwargs
    ) -> str:
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
                # To ensure the only one system prompt
                if self._multi_turn_system_prompt_already_set and system_prompt:
                    if self.__first_turn:
                        log(
                            "Since the initial system prompt was already set, the current system prompt is ignored.",
                            "warning",
                        )
                        self.__first_turn = False
                    system_prompt = None

                # System prompt
                if system_prompt:
                    self.histories.append({"role": "system", "content": system_prompt})
                    self._multi_turn_system_prompt_already_set = True

                # User prompt
                self.histories += self.__make_payload(user_prompt)

                # Model response
                response = self.client.chat.completions.create(
                    model=self.model, messages=self.histories, **kwargs
                )
                assistant_msg = response.choices[0].message
                self.histories.append(
                    {"role": assistant_msg.role, "content": assistant_msg.content}
                )

            else:
                # To ensure empty history
                self.reset_history()

                # System prompt
                payloads = [{"role": "system", "content": system_prompt}] if system_prompt else []

                # User prompt
                payloads += self.__make_payload(user_prompt)

                # Model response
                response = self.client.chat.completions.create(
                    model=self.model, messages=payloads, **kwargs
                )
                assistant_msg = response.choices[0].message

            return assistant_msg.content

        except Exception as e:
            raise e
