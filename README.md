# PatientSim-pkg

---
![Python Versions](https://img.shields.io/badge/python-3.11%2B%2C%203.12%2B-blue)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/patientsim)
![PyPI Version](https://img.shields.io/pypi/v/patientsim)
![Downloads](https://img.shields.io/pypi/dm/patientsim)
![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2505.17818-blue)
---

An official Python package for simulating patient interactions, called `PatientSim`.

&nbsp;


### Recent updates 📣
* *August 2025 (v0.1.2)*: Added support for emergency department simulation, Azure for GPT, and Vertex AI for the Gemini API.
* *August 2025 (v0.1.1)*: Added support for a doctor persona in the LLM agent for the emergency department.
* *August 2025 (v0.1.0)*: Initial release: Introduced a dedicated LLM agent for patients that allows customization of patient personas.

&nbsp;

&nbsp;

## Installation
```bash
pip install patientsim
```
```python
import patientsim
print(patientsim.__version__)
```

&nbsp;

&nbsp;


## Overview 📚
*This repository is the official repository for the PyPI package. For the repository related to the paper and experiments, please refer to [here](https://anonymous.4open.science/r/PatientSim-2691/README.md).*

&nbsp;

&nbsp;



## Quick Starts 🚀
*If you plan to run this simulation with real clinical data or other sensitive information, you must use Vertex AI (for Gemini) or Azure OpenAI (for GPT).
When using Azure OpenAI, be sure to opt out of human review of the data to maintain compliance and ensure privacy protection.*

> [!NOTE]
> Before using the LLM API, you must provide the API key for each model directly or specify it in a `.env` file.
> * *gemini-\**: If you set the model to a Gemini LLM, you must have your own GCP API key in the `.env` file, with the name `GOOGLE_API_KEY`. The code will automatically communicate with GCP.
>* *gpt-\**: If you set the model to a GPT LLM, you must have your own OpenAI API key in the `.env` file, with the name `OPENAI_API_KEY`. The code will automatically use the OpenAI chat format.

> [!NOTE]
> To use Vertex AI, you must complete the following setup steps:
> 1) Select or create a Google Cloud project in the Google Cloud Console.
> 2) Enable the Vertex AI API.
> 3) Create a Service Account:
>    * Navigate to **IAM & Admin > Service Accounts**
>    * Click **Create Service Account**
>    * Assign the role **Vertex AI Platform Express User**
> 4. Generate a credential key in JSON format and set the path to this JSON file in the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

&nbsp;

### Environment Variables
Before using the LLM API, you need to provide the API key (or the required environment variables for each model) either directly or in a .env file.
```bash
# For GPT API without Azure
OPENAI_API_KEY="YOUR_OPENAI_KEY"

# For GPT API with Azure
AZURE_ENDPOINT="https://your-azure-openai-endpoint"

# For Gemini API without Vertex AI
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY" 

# For Gemini API with Vertex AI
GOOGLE_PROJECT_ID="your-gcp-project-id"
GOOGLE_PROJECT_LOCATION="your-gcp-project-location"  # (e.g., us-central1)
GOOGLE_APPLICATION_CREDENTIALS="/path/to/google_credentials.json" # Path to GCP service account credentials (JSON file)
```

&nbsp;

### Agent Initialization
**Patient Agent**
```python
# Patient Agent (gpt)
from patientsim import PatientAgent
patient_agent = PatientAgent('gpt-4o', 
                              visit_type='emergency_department',
                              random_seed=42,
                              random_sampling=False,
                              temperature=0,
                              api_key=OPENAI_API_KEY,
                              use_azure=False   # Set True if using Azure
                            )

# Patient Agent (gemini)
patient_agent = PatientAgent('gemini-2.5-flash', 
                              visit_type='emergency_department',
                              random_seed=42,
                              random_sampling=False,
                              temperature=0,
                              api_key=GOOGLE_API_KEY,
                              use_vertex=False # Set True for use Vertex AI
                            )

response = patient_agent(
    user_prompt="How can I help you?",
)

print(response)

# Example response:
# > I'm experiencing some concerning symptoms, but I can't recall any specific medical history.
# > You are playing the role of a kind and patient doctor...
```

**Doctor Agent**
```python
doctor_agent = DoctorAgent('gpt-4o', use_azure=False)
doctor_agent = DoctorAgent('gemini-2.5-flash', use_vertex=False)
print(doctor_agent.system_prompt)
```

&nbsp;

### Run Emergency Department Simulation
```python
simulation_env = EDSimulation(patient_agent, doctor_agent)
simulation_env.simulate()

# Example response:
# Example response:
# > Doctor   [0%]  : Hello, how can I help you?
# > Patient  [6%]  : I'm experiencing some concerning symptoms,
# > Doctor   [6%]  : I'm sorry to hear that you're experiencing difficulty. When dit this start?
# > Patient  [13%] : Three hours prior to my arrival.
# > ...
```



&nbsp;

&nbsp;


## Citation
```
@misc{kyung2025patientsimpersonadrivensimulatorrealistic,
      title={PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions}, 
      author={Daeun Kyung and Hyunseung Chung and Seongsu Bae and Jiho Kim and Jae Ho Sohn and Taerim Kim and Soo Kyung Kim and Edward Choi},
      year={2025},
      eprint={2505.17818},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.17818}, 
}
```

&nbsp;