import os
import random
from typing import Optional
from importlib import resources

from patientsim.registry.persona import *
from patientsim.utils import colorstr, log
from patientsim.utils.desc_utils import *
from patientsim.utils.common_utils import *
from patientsim.client import GeminiClient, GeminiVertexClient, GPTClient, GPTAzureClient



class PatientAgent:
    def __init__(self,
                 model: str,
                 visit_type: str = 'emergency_department',
                 personality: str = 'plain',
                 recall_level: str = 'no_history',
                 confusion_level: str = 'normal',
                 lang_proficiency_level: str = 'C',
                 api_key: Optional[str] = None,
                 use_azure: bool = False,
                 use_vertex: bool = False,
                 azure_endpoint: Optional[str] = None,
                 genai_project_id: Optional[str] = None,
                 genai_project_location: Optional[str] = None,
                 genai_credential_path: Optional[str] = None,
                 system_prompt_path: Optional[str] = None,
                 **kwargs) -> None:
        
        # Initialize patient attributes
        self.visit_type = visit_type.lower()
        self.personality = personality.lower()
        self.recall_level = recall_level.lower()
        self.confusion_level = confusion_level.lower()
        self.lang_proficiency_level = lang_proficiency_level.upper()
        self.__sanity_check()

        # Initialize environment
        self._init_env(**kwargs)
        
        # Initialize model, API client, and other parameters
        self.model = model
        self._init_model(self.model, api_key, use_azure, use_vertex, azure_endpoint, genai_project_id, genai_project_location, genai_credential_path)
        
        # Initialize prompt
        system_prompt_template = self._init_prompt(self.visit_type, system_prompt_path)
        self.system_prompt = self.build_prompt(system_prompt_template)
        
        log("PatientAgent initialized successfully", color=True)
    

    def _init_env(self, **kwargs) -> None:
        """
        Initialize the environment with default settings.
        """
        self.random_seed = kwargs.get('random_seed', None)
        # Set random seed for reproducibility
        if self.random_seed:
            set_seed(self.random_seed)
        self.temperature = kwargs.get('temperature', 0.2)   # For various responses. If you want deterministic responses, set it to 0.
        self.num_word_sample = kwargs.get('num_word_sample', 3)
        self.random_sampling = kwargs.get('random_sampling', True)
        self._terms = {
            'cefr_A1': split_string(kwargs.get('cefr_A1', [])),
            'cefr_A2': split_string(kwargs.get('cefr_A2', [])),
            'cefr_B1': split_string(kwargs.get('cefr_B1', [])),
            'cefr_B2': split_string(kwargs.get('cefr_B2', [])),
            'cefr_C1': split_string(kwargs.get('cefr_C1', [])),
            'cefr_C2': split_string(kwargs.get('cefr_C2', [])),
            'med_A': split_string(kwargs.get('med_A', [])),
            'med_B': split_string(kwargs.get('med_B', [])),
            'med_C': split_string(kwargs.get('med_C', [])),
        }
        self.patient_conditions = {
            'name': kwargs.get('name', 'James Lee'),
            'birth_date': kwargs.get('birth_date', generate_random_date()),
            'age': kwargs.get('age', str(random.randint(20, 80))),
            'gender': kwargs.get('gender', random.choice(['male', 'female'])),
            'race': kwargs.get('race', 'N/A'),
            'tobacco': kwargs.get('tobacco', 'N/A'),
            'alcohol': kwargs.get('alcohol', 'N/A'),
            'illicit_drug': kwargs.get('illicit_drug', 'N/A'),
            'exercise': kwargs.get('exercise', 'N/A'),
            'marital_status': kwargs.get('marital_status', 'N/A'),
            'children': kwargs.get('children', 'N/A'),
            'living_situation': kwargs.get('living_situation', 'N/A'),
            'occupation': kwargs.get('occupation', 'N/A'),
            'insurance': kwargs.get('insurance', 'N/A'),
            'allergies': kwargs.get('allergies', 'N/A'),
            'family_medical_history': kwargs.get('family_medical_history', 'N/A'),
            'medical_device': kwargs.get('medical_device', 'N/A'),
            'medical_history': kwargs.get('medical_history', 'N/A'),
            'present_illness_positive': kwargs.get('present_illness_positive', 'N/A'),
            'present_illness_negative': kwargs.get('present_illness_negative', 'N/A'),
            'chiefcomplaint': kwargs.get('chiefcomplaint', 'N/A'),
            'pain': kwargs.get('pain', 'N/A'),
            'medication': kwargs.get('medication', 'N/A'),
            'arrival_transport': kwargs.get('arrival_transport', 'N/A'),
            'disposition': kwargs.get('disposition', 'N/A'),
            'diagnosis': kwargs.get('diagnosis', 'N/A'),
            'department': kwargs.get('department', None),
            'symptom': kwargs.get('symptom', None),
        }

        if self.visit_type == 'outpatient':
            assert self.patient_conditions.get('department'), \
                log(colorstr("red", "To simulate outpatient, you should provide a specific department."))
            assert self.patient_conditions.get('symptom'), \
                log(colorstr("red", "To simulate outpatient, you should provide at least a simple symptom."))
        

    def _init_model(self,
                    model: str,
                    api_key: Optional[str] = None,
                    use_azure: bool = False,
                    use_vertex: bool = False,
                    azure_endpoint: Optional[str] = None,
                    genai_project_id: Optional[str] = None,
                    genai_project_location: Optional[str] = None,
                    genai_credential_path: Optional[str] = None) -> None:
        """
        Initialize the model and API client based on the specified model type.

        Args:
            model (str): The patient agent model to use.
            api_key (Optional[str], optional): API key for the model. If not provided, it will be fetched from environment variables.
                                               Defaults to None.
            use_azure (bool): Whether to use Azure OpenAI client.
            use_vertex (bool): Whether to use Google Vertex AI client.
            azure_endpoint (Optional[str], optional): Azure OpenAI endpoint. Defaults to None.
            genai_project_id (Optional[str], optional): Google Cloud project ID. Defaults to None.
            genai_project_location (Optional[str], optional): Google Cloud project location. Defaults to None.
            genai_credential_path (Optional[str], optional): Path to Google Cloud credentials JSON file. Defaults to None.

        Raises:
            ValueError: If the specified model is not supported.
        """
        if 'gemini' in self.model.lower():
            self.client = GeminiVertexClient(model, genai_project_id, genai_project_location, genai_credential_path) if use_vertex else GeminiClient(model, api_key)
        elif 'gpt' in self.model.lower():       # TODO: Support o3, o4 models etc.
            self.client = GPTAzureClient(model, api_key, azure_endpoint) if use_azure else GPTClient(model, api_key)
        else:
            raise ValueError(colorstr("red", f"Unsupported model: {self.model}. Supported models are 'gemini' and 'gpt'."))
        

    def _init_prompt(self, visit_type: str, system_prompt_path: Optional[str] = None) -> str:
        """
        Initialize the system prompt for the patient agent.

        Args:
            visit_type (str): Type of visit, either 'outpatient' or 'emergency_department'.
            system_prompt_path (Optional[str], optional): Path to a custom system prompt file. 
                                                          If not provided, the default system prompt will be used. Defaults to None.

        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Initialilze with the default system prompt
        if not system_prompt_path:
            if visit_type == 'outpatient':
                prompt_file_name = "op_patient_sys.txt"     # TODO: Make outpatient system prompt
            else:
                prompt_file_name = "ed_uti_patient_sys.txt" if self.patient_conditions.get('diagnosis').lower() == 'urinary tract infection' else "ed_patient_sys.txt"
            file_path = resources.files("patientsim.assets.prompt").joinpath(prompt_file_name)
            system_prompt = file_path.read_text()
        
        # User can specify a custom system prompt
        else:
            if not os.path.exists(system_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {system_prompt_path}"))
            with open(system_prompt_path, 'r') as f:
                system_prompt = f.read()
        return system_prompt


    def __sanity_check(self) -> None:
        """
        Sanity check for patient attributes to ensure they are valid.

        Raises:
            ValueError: If any of the attributes are not valid.
        """
        if self.personality not in PERSONALITY:
            raise ValueError(colorstr("red", f"Invalid personality type: {self.personality}. Supported types: {', '.join(PERSONALITY.keys())}"))
        if self.recall_level not in RECALL_LEVEL:
            raise ValueError(colorstr("red", f"Invalid recall level: {self.recall_level}. Supported levels: {', '.join(RECALL_LEVEL.keys())}"))
        if self.confusion_level not in CONFUSION_LEVEL:
            raise ValueError(colorstr("red", f"Invalid confusion level: {self.confusion_level}. Supported levels: {', '.join(CONFUSION_LEVEL.keys())}"))
        if self.lang_proficiency_level not in LANG_PROFICIENCY_LEVEL:
            raise ValueError(colorstr("red", f"Invalid language proficiency level: {self.lang_proficiency_level}. Supported levels: {', '.join(LANG_PROFICIENCY_LEVEL.keys())}"))
        if self.visit_type not in VISIT_TYPE:
            raise ValueError(colorstr("red", f"Invalid visiting type: {self.visit_type}. Supported types: {', '.join(VISIT_TYPE)}"))
        
    
    def build_prompt(self, system_prompt_template: str) -> str:
        """
        Build the system prompt for the patient agent.

        Args:
            system_prompt_template (str): The template for the system prompt.

        Returns:
            str: The formatted system prompt with patient attributes filled in.
        """
        personality_desc = get_personality_description(self.personality)
        recall_desc = get_recall_description(self.recall_level)
        confusion_desc = get_confusion_description(self.confusion_level)
        lang_proficiency_desc = get_language_proficiency_description(
            self.lang_proficiency_level,
            num_sample=self.num_word_sample,
            random_sampling=self.random_sampling,
            **self._terms
        )
        reminder_desc = get_reminder_description(
            self.personality, 
            self.lang_proficiency_level, 
            self.recall_level, 
            self.confusion_level
        )
        sentence_limit = PERSONALITY_SENTENCE_LIMIT.get(self.personality, "3")
        system_prompt = system_prompt_template.format(
            personality=personality_desc,
            recall=recall_desc,
            confusion=confusion_desc,
            lang_proficiency=lang_proficiency_desc,
            reminder=reminder_desc,
            sentence_limit=sentence_limit,
            **self.patient_conditions
        )
        prompt_valid_check(system_prompt, self.patient_conditions)
        return system_prompt
    

    def __call__(self,
                 user_prompt: str,
                 using_multi_turn: bool = True,
                 verbose: bool = True) -> str:
        """
        Call the patient agent with a user prompt and return the response.

        Args:
            user_prompt (str): The user prompt to send to the patient agent.
            using_multi_turn (bool, optional): Whether to use multi-turn conversation. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            str: The response from the patient agent.
        """
        response = self.client(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            using_multi_turn=using_multi_turn,
            verbose=verbose,
            temperature=self.temperature,
            seed=self.random_seed
        )
        return response
        