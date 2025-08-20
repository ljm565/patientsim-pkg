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

### Environment Variables
```python
# OpenAI / Azure (Linux/Mac)
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export AZURE_ENDPOINT="https://your-azure-openai-endpoint"

# Google Vertex / Gemini (Linux/Mac)
export GOOGLE_PROJECT_ID="your-gcp-project-id"
export GOOGLE_PROJECT_LOCATION="us-central1"   # example
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/google_credentials.json"
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"    # if not using Vertex

# Windows (PowerShell example)
# setx OPENAI_API_KEY "YOUR_OPENAI_KEY"
# setx AZURE_ENDPOINT "https://your-azure-openai-endpoint"
# setx GOOGLE_PROJECT_ID "your-gcp-project-id"
# setx GOOGLE_PROJECT_LOCATION "us-central1"
# setx GOOGLE_APPLICATION_CREDENTIALS "C:\path\to\service_account.json"
# setx GEMINI_API_KEY "YOUR_GEMINI_API_KEY"
```



### Agent Initialization
**Patient Agent**
```python
# Patient Agent (gpt + azure)
from patientsim import PatientAgent
patient_agent = PatientAgent('gpt-4o', 
                              visit_type='emergency_department',
                              random_seed=42,
                              random_sampling=False,
                              temperature=0,
                              api_key=OPENAI_API_KEY,
                              use_azure=True, #  set False if using OpenAI directly
                              azure_endpoint=AZURE_ENDPOINT # required for Azure
                            )

response = patient_agent(
    user_prompt="How can I help you?",
)

print(response)

# Patient Agent (gemini)
# Use Vertex AI
patient_agent = PatientAgent('gemini-2.5-flash', 
                              visit_type='emergency_department',
                              random_seed=42,
                              random_sampling=False,
                              temperature=0,
                              use_vertex=True,
                              genai_project_id=GOOGLE_PROJECT_ID,
                              genai_project_location=GOOGLE_PROJECT_LOCATION,
                              genai_credential_path=GOOGLE_APPLICATION_CREDENTIALS)
                    
# Use Gemini API directly          
patient_agent = PatientAgent('gemini-2.5-flash', 
                              visit_type='emergency_department',
                              random_seed=42,
                              random_sampling=False,
                              temperature=0,
                              use_vertex=False,
                              api_key=GEMINI_API_KEY)

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
doctor_agent = DoctorAgent('gpt-4o', 
                            api_key=OPENAI_API_KEY,
                            use_azure=True, #  set False if using OpenAI directly
                            azure_endpoint=AZURE_ENDPOINT # required for Azure
                        )

doctor_agent = DoctorAgent('gemini-2.5-flash', 
                            api_key=GEMINI_API_KEY,
                            use_vertex=True, #  set False if using OpenAI directly
                            genai_project_id=GOOGLE_PROJECT_ID,
                            genai_project_location=GOOGLE_PROJECT_LOCATION,
                            genai_credential_path=GOOGLE_APPLICATION_CREDENTIALS
                        )
print(doctor_agent.system_prompt)
```


### Run ED Simulation
```python
simulation_env = EDSimulation(patient_agent, doctor_agent)
simulation_env.simulate()

# Example response:
# > Doctor   [0%]  : Hello, how can I help you?
# > Patient  [6%]  : I'm experiencing some concerning symptoms,
# > Doctor   [6%]  : Hello, how can I help you?
# > Patient  [13%]  : I'm experiencing some concerning symptoms,
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