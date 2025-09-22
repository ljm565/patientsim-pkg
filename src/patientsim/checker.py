import os
from typing import Optional
from importlib import resources

from patientsim.registry.persona import *
from patientsim.utils import colorstr, log
from patientsim.utils.desc_utils import *
from patientsim.utils.common_utils import *
from patientsim.client import GeminiClient, GeminiVertexClient, GPTClient, GPTAzureClient, VLLMClient



class CheckerAgent:
    def __init__(self,
                 model: str,
                 visit_type: str = 'emergency_department',
                 api_key: Optional[str] = None,
                 use_azure: bool = False,
                 use_vertex: bool = False,
                 use_vllm: bool = False,
                 azure_endpoint: Optional[str] = None,
                 genai_project_id: Optional[str] = None,
                 genai_project_location: Optional[str] = None,
                 genai_credential_path: Optional[str] = None,
                 vllm_endpoint: Optional[str] = None,
                 user_prompt_path: Optional[str] = None,
                 **kwargs) -> None:
        
        # Initialize attributes
        self.visit_type = visit_type.lower()
        self.__sanity_check()
        
        # Initialize model, API client, and other parameters
        self.model = model
        self.random_seed = kwargs.get('random_seed', None)
        self.temperature = kwargs.get('temperature', 0.0)   # Use 0.0 during evaluation for deterministic (non-random) responses.
        self._init_model(
            model=self.model,
            api_key=api_key,
            use_azure=use_azure,
            use_vertex=use_vertex,
            use_vllm=use_vllm,
            azure_endpoint=azure_endpoint,
            genai_project_id=genai_project_id,
            genai_project_location=genai_project_location,
            genai_credential_path=genai_credential_path,
            vllm_endpoint=vllm_endpoint
        )
        
        # Initialize prompt
        self.prompt_template = self._init_prompt(self.visit_type, user_prompt_path)
        
        log("CheckerAgent initialized successfully", color=True)
    
        
    def _init_model(self,
                    model: str,
                    api_key: Optional[str] = None,
                    use_azure: bool = False,
                    use_vertex: bool = False,
                    use_vllm: bool = False,
                    azure_endpoint: Optional[str] = None,
                    genai_project_id: Optional[str] = None,
                    genai_project_location: Optional[str] = None,
                    genai_credential_path: Optional[str] = None,
                    vllm_endpoint: Optional[str] = None) -> None:
        """
        Initialize the model and API client based on the specified model type.

        Args:
            model (str): The checker agent model to use.
            api_key (Optional[str], optional): API key for the model. If not provided, it will be fetched from environment variables.
                                               Defaults to None.
            use_azure (bool): Whether to use Azure OpenAI client.
            use_vertex (bool): Whether to use Google Vertex AI client.
            use_vllm (bool): Whether to use vLLM client.
            azure_endpoint (Optional[str], optional): Azure OpenAI endpoint. Defaults to None.
            genai_project_id (Optional[str], optional): Google Cloud project ID. Defaults to None.
            genai_project_location (Optional[str], optional): Google Cloud project location. Defaults to None.
            genai_credential_path (Optional[str], optional): Path to Google Cloud credentials JSON file. Defaults to None.
            vllm_endpoint (Optional[str], optional): Path to the vLLM server. Defaults to None.

        Raises:
            ValueError: If the specified model is not supported.
        """
        if 'gemini' in self.model.lower():
            self.client = GeminiVertexClient(model, genai_project_id, genai_project_location, genai_credential_path) if use_vertex else GeminiClient(model, api_key)
        elif 'gpt' in self.model.lower():       # TODO: Support o3, o4 models etc.
            self.client = GPTAzureClient(model, api_key, azure_endpoint) if use_azure else GPTClient(model, api_key)
        elif use_vllm:
            self.client = VLLMClient(model, vllm_endpoint)
        else:
            raise ValueError(colorstr("red", f"Unsupported model: {self.model}. Supported models are 'gemini' and 'gpt'."))
        

    def _init_prompt(self, visit_type: str, user_prompt_path: Optional[str] = None) -> str:
        """
        Initialize the user prompt for the checker agent.

        Args:
            visit_type (str): Type of visit, either 'outpatient' or 'emergency_department'.
            user_prompt_path (Optional[str], optional): Path to a custom user prompt file. 
                                                          If not provided, the default user prompt will be used. Defaults to None.

        Raises:
            FileNotFoundError: If the specified user prompt file does not exist.
        """
        # Initialilze with the default user prompt
        if not user_prompt_path:
            if visit_type == 'outpatient':
                prompt_file_name = "op_terminate_user.txt"
            else:
                prompt_file_name = "ed_terminate_user.txt"
            file_path = resources.files("patientsim.assets.prompt").joinpath(prompt_file_name)
            user_prompt = file_path.read_text()
        
        # User can specify a custom user prompt
        else:
            if not os.path.exists(user_prompt_path):
                raise FileNotFoundError(colorstr("red", f"User prompt file not found: {user_prompt_path}"))
            with open(user_prompt_path, 'r') as f:
                user_prompt = f.read()
        return user_prompt


    def __sanity_check(self) -> None:
        """
        Sanity check for checker attributes to ensure they are valid.

        Raises:
            ValueError: If any of the attributes are not valid.
        """
        if self.visit_type not in VISIT_TYPE:
            raise ValueError(colorstr("red", f"Invalid visiting type: {self.visit_type}. Supported types: {', '.join(VISIT_TYPE)}"))


    def __call__(self,
                 response: str) -> str:
        """
        Call the checker agent with a response, and then return the result.

        Args:
            response (str): Target response to evaluate.

        Returns:
            str: The response from the checker agent.
        """
        response = self.client(
            user_prompt=self.prompt_template.format(response=response),
            using_multi_turn=False,
            verbose=False,
            temperature=self.temperature,
            seed=self.random_seed,
        )
        return response
        