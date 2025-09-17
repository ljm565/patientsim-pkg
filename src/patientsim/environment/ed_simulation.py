import os
import time

from typing import Optional
from importlib import resources
from patientsim.doctor import DoctorAgent
from patientsim.patient import PatientAgent
from patientsim.utils import log, colorstr
from patientsim.utils.common_utils import detect_ed_termination
from patientsim.client import GeminiClient, GeminiVertexClient, GPTClient, GPTAzureClient


class EDSimulation:
    def __init__(
        self,
        patient_agent: PatientAgent,
        doctor_agent: DoctorAgent,
        max_inferences: int = 15,
        enable_llm_termination_check: bool = False,
        termination_checker: str = "gemini-2.5-flash",
        termination_checker_prompt_path: Optional[str] = None,
        api_key: Optional[str] = None,
        use_azure: bool = False,
        use_vertex: bool = False,
        azure_endpoint: Optional[str] = None,
        genai_project_id: Optional[str] = None,
        genai_project_location: Optional[str] = None,
        genai_credential_path: Optional[str] = None,
    ):

        # Initialize simulation parameters
        self.patient_agent = patient_agent
        self.doctor_agent = doctor_agent
        self.max_inferences = max_inferences
        self.current_inference = 0  # Current inference index
        self.enable_llm_termination_check = enable_llm_termination_check  # Flag for LLM-based termination detection
        self.termination_checker = termination_checker
        if self.enable_llm_termination_check:
            log("LLM-based termination detection is enabled.", level="warning")
            self._init_termination_checker(
                termination_checker,
                termination_checker_prompt_path,
                api_key,
                use_azure,
                use_vertex,
                azure_endpoint,
                genai_project_id,
                genai_project_location,
                genai_credential_path,
            )

        self._sanity_check()

    def _sanity_check(self):
        """
        Verify and synchronize the maximum number of inference rounds 
        between the Doctor agent and the ED simulation.

        If the configured values do not match, a warning is logged and 
        the Doctor agent's configuration is updated to align with the 
        ED simulation. The system prompt is also rebuilt accordingly.
        """
        if not self.doctor_agent.max_inferences == self.max_inferences:
            log("The maximum number of inferences between the Doctor agent and the ED simulation does not match.", level="warning")
            log(f"The simulation will start with the value ({self.max_inferences}) configured in the ED simulation, \
                and the Doctor agent system prompt will be updated accordingly.", level="warning")
            self.doctor_agent.max_inferences = self.max_inferences
            self.doctor_agent.build_prompt()

    def _init_termination_checker(
        self,
        model: str,
        prompt_path: Optional[str],
        api_key: Optional[str],
        use_azure: bool,
        use_vertex: bool,
        azure_endpoint: Optional[str],
        genai_project_id: Optional[str],
        genai_project_location: Optional[str],
        genai_credential_path: Optional[str],
    ):
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

        # Initialize the model and API client
        if "gemini" in self.termination_checker.lower():
            self.termination_checker_client = (
                GeminiVertexClient(model, genai_project_id, genai_project_location, genai_credential_path) if use_vertex else GeminiClient(model, api_key)
            )
        elif "gpt" in self.termination_checker.lower():  # TODO: Support o3, o4 models etc.
            self.termination_checker_client = GPTAzureClient(model, api_key, azure_endpoint) if use_azure else GPTClient(model, api_key)
        else:
            raise ValueError(colorstr("red", f"Unsupported model: {self.model}. Supported models are 'gemini' and 'gpt'."))

        # Initialilze with the default system prompt
        if not prompt_path:
            file_path = resources.files("patientsim.assets.prompt").joinpath("ed_terminate_user.txt")
            termination_checker_prompt = file_path.read_text()
        else:
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(colorstr("red", f"Prompt file for termination model not found: {prompt_path}"))
            with open(prompt_path, "r") as f:
                termination_checker_prompt = f.read()
        self.termination_checker_prompt = termination_checker_prompt

    def simulate(self, verbose: bool = True) -> list[dict]:
        """
        Run a full conversation simulation between the Doctor and Patient agents
        in the emergency department setting.

        The simulation alternates turns between the Doctor and Patient until
        the maximum number of inference rounds is reached or early termination
        is detected. During the final turn, the Doctor agent is instructed to
        provide its top 5 differential diagnoses.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        
        Returns:
            list[dict]: Dialogue history, where each dict contains:
                - "role" (str): "Doctor" or "Patient"
                - "content" (str): Text content of the dialogue turn
        """
        if verbose:
            log(f"Patient prompt:\n{self.patient_agent.system_prompt}")
            log(f"Doctor prompt:\n{self.doctor_agent.system_prompt}")

        # Start conversation
        doctor_greet = self.doctor_agent.doctor_greet
        dialog_history = [{"role": "Doctor", "content": doctor_greet}]
        role = f"{colorstr('blue', 'Doctor')}  [0%]"
        log(f"{role:<23}: {doctor_greet}")

        for inference_idx in range(self.max_inferences):
            progress = int(((inference_idx + 1) / self.max_inferences) * 100)

            # Obtain response from patient
            patient_response = self.patient_agent(
                user_prompt=dialog_history[-1]["content"],
                using_multi_turn=True,
                verbose=verbose
            )
            dialog_history.append({"role": "Patient", "content": patient_response})
            role = f"{colorstr('green', 'Patient')} [{progress}%]"
            log(f"{role:<23}: {patient_response}")

            # Obtain response from doctor
            doctor_response = self.doctor_agent(
                user_prompt=dialog_history[-1]["content"] + "\nThis is the final turn. Now, you must provide your top5 differential diagnosis." \
                    if inference_idx == self.max_inferences - 1 else dialog_history[-1]["content"],
                using_multi_turn=True,
                verbose=verbose
            )
            dialog_history.append({"role": "Doctor", "content": doctor_response})
            role = f"{colorstr('blue', 'Doctor')}  [{progress}%]"
            log(f"{role:<23}: {doctor_response}")

            # If early termination is detected, break the loop
            if detect_ed_termination(doctor_response):
                break

            elif self.enable_llm_termination_check:
                # Prepare the prompt for termination detection
                termination_prompt = self.termination_checker_prompt.format(response=doctor_response)

                # Get the termination model's response
                termination_checker_response = self.termination_checker_client(user_prompt=termination_prompt, using_multi_turn=False, verbose=False).strip().upper()

                # Check if the response indicates termination
                if termination_checker_response == "Y":
                    log("Consultation termination detected by the LLM-based checker.", level="warning")
                    break

            # Prevent API timeouts
            time.sleep(1.0)

        log("Simulation completed.", color=True)

        return dialog_history
