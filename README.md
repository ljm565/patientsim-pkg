# PatientSim-pkg

---
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/patientsim)
![PyPI Version](https://img.shields.io/pypi/v/patientsim)
![Downloads](https://img.shields.io/pypi/dm/patientsim)
![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2505.17818-blue)
---

An official Python package for simulating patient interactions, called `PatientSim`.
By setting a patient persona and assigning it to the LLM agent, you can generate outcomes of interactions with a doctor.
The patient persona consists of four elements, resulting in 37 unique combinations:
* Personality: `plain` (default), `verbose`, `pleasing`, `impatient`, `distrust`, `overanxious`.
* Language Proficiency: `C` (default), `B`, `A` (C means the highest level).
* Medical History Recall Level: `no_history` (default), `low`, `high`.
* Cognitive Confusion Level: `normal` (default), `moderate`, `high`.

The simulation scenarios also have two visit types:
* Outpatient: `outpatient`
* Emergency: `emergency_department`

&nbsp;

### Recent updates 📣
* *September 2025 (v0.1.7)*: Fixed typos of the prompts.
* *September 2025 (v0.1.6)*: Updated dependencies.
* *August 2025 (v0.1.5)*: Improved the outpatient simulation to be more realistic based on expert feedback.
* *August 2025 (v0.1.4)*: Added support for outpatient simulation and added exception handling for None-type responses from Gemini.
* *August 2025 (v0.1.3)*: Added support for emergency department simulation, Azure for GPT, and Vertex AI for the Gemini API.
* *August 2025 (v0.1.1)*: Added support for a doctor persona in the LLM agent for the emergency department.
* *August 2025 (v0.1.0)*: Initial release: Introduced a dedicated LLM agent for patients that allows customization of patient personas.

&nbsp;

&nbsp;

## Installation 🛠️
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
*This repository is the official repository for the [PyPI package](https://pypi.org/project/patientsim/).For the repository related to the paper and experiments, please refer to [here](https://github.com/dek924/PatientSim).*

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
#### Patient Agent
1. Default settings usage.
```python
# Patient Agent (GPT)
from patientsim import PatientAgent

patient_agent = PatientAgent('gpt-4o', 
                              visit_type='emergency_department',
                              random_seed=42,
                              random_sampling=False,
                              temperature=0,
                              api_key=OPENAI_API_KEY,
                              use_azure=False   # Set True if using Azure
                            )

# Patient Agent (Gemini)
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

&nbsp;

2. Apply custom persona.
```python
from patientsim import PatientAgent

patient_agent = PatientAgent('gpt-4o', 
                              visit_type='emergency_department',
                              personality='verbose',
                              recall_level='low',
                              confusion_level='moderate',
                              lang_proficiency_level='B',
                              age='45',
                              tobacco='Denies tobacco use',
                              allergies="Penicillins",
                              ...
                            )
```
> Persona Arguments:
> * `visit_type` (str): `emergency_department` (default), `outpatient`
> * `personality` (str): `plain` (default), `verbose`, `pleasing`, `impatient`, `distrust`, `overanxious`
> * `recall_level` (str): `no_history` (default), `low`, `high`
> * `confusion_level` (str): `normal` (default), `moderate`, `high`
> * `lang_proficiency_level`: `C` (default), `B`, `A` (C means the highest level).
> * `name` (str): Patient's name. Default: "James Lee".
> * `birth_date` (str): Patient's birth_date. Default: random date between 1960-01-01 and 2000-12-31.
> * `age` (str): Patient's age. Default: random.randint(20, 80). he value is randomly generated and does not depend on the birth date.
> * `gender` (str): Patient's gender. Default: random.choice(['male', 'female']).
> * `race` (str): Patient's race or ethnicity. Default: "N/A".
> * `tobacco` (str): Patient's tobacco use status (e.g., current, former, never). Default: "N/A".
> * `alcohol` (str): Patient's alcohol use status (e.g., current, former, never). Default: "N/A".
> * `illicit_drug` (str): Patient's illicit drug use status. Default: "N/A".
> * `exercise` (str): Patient's physical activity level or exercise habits. Default: "N/A".
> * `marital_status` (str): Patient's marital status (e.g., single, married, divorced). Default: "N/A".
> * `children` (str): Number of children or information about dependents. Default: "N/A".
> * `living_situation` (str): Patient's current living arrangement (e.g., alone, with family). Default: "N/A".
> * `occupation` (str): Patient's occupation or job information. Default: "N/A".
> * `insurance` (str): Patient's health insurance status or type. Default: "N/A".
> * `allergies` (str): Known allergies of the patient (medication, food, environmental). Default: "N/A".
> * `family_medical_history` (str): Relevant medical history of the patient's family. Default: "N/A".
> * `medical_device` (str): Any medical devices the patient uses (e.g., pacemaker, insulin pump). Default: "N/A".
> * `medical_history` (str): Patient's past medical history (conditions, surgeries, hospitalizations). Default: "N/A".
> * `present_illness_positive` (str): Positive symptoms or findings for the current illness. Default: "N/A".
> * `present_illness_negative` (str): Negative symptoms or findings for the current illness. Default: "N/A".
> * `chiefcomplaint` (str): Main reason the patient seeks medical attention. Default: "N/A".
> * `pain` (str): Description or severity of pain, if any. Default: "N/A".
> * `medication` (str): Current medications the patient is taking. Default: "N/A".
> * `arrival_transport` (str): How the patient arrived at the facility (e.g., ambulance, private vehicle). Default: "N/A".
> * `disposition` (str): Planned disposition after evaluation (e.g., discharge, admission). Default: "N/A".
> * `diagnosis` (str): Diagnosed condition(s) for the patient. Default: "N/A".

&nbsp;

#### Doctor Agent
```python
from patientsim import DoctorAgent

doctor_agent = DoctorAgent('gpt-4o', use_azure=False)
doctor_agent = DoctorAgent('gemini-2.5-flash', use_vertex=False)
print(doctor_agent.system_prompt)
```

&nbsp;

### Run Emergency Department Simulation
```python
from patientsim.environment import EDSimulation

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