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
```python
# Make sure to set your OpenAI API key first
# export OPENAI_API_KEY="YOUR_KEY"  (Linux/Mac)
# setx OPENAI_API_KEY "YOUR_KEY"     (Windows)

# Patient Agent (gpt + azure)
patient_agent = PatientAgent('gpt-4o', 
                              visit_type='emergency_department',
                              random_seed=42,
                              random_sampling=False,
                              temperature=0,
                              api_key=OPENAI_API_KEY,
                              use_azure=True,
                              azure_endpoint=AZURE_ENDPOINT)

response = patient_agent(
    user_prompt="How can I help you?",
)

print(response)

# Patient Agent (gemini + vertex)
# Before use vertex ai
# 1) Select or create a Google Cloud project in the Google Cloud Console.
# 2) Enable the Vertex AI API.
# 3) Create a service account:
#       1. Go to IAM & Admin > Service Accounts
#       2. Click Create service account
#       3. Assign the Vertex AI Platform Express User role
# 4) Generate a key in JSON format and set the path to the JSON file as the GOOGLE_APPLICATION_CREDENTIALS environment variable.

patient_agent = PatientAgent('gemini-2.5-flash', 
                              visit_type='emergency_department',
                              random_seed=42,
                              random_sampling=False,
                              temperature=0,
                              use_vertex=True,
                              genai_project_id=GOOGLE_PROJECT_ID,
                              genai_project_location=GOOGLE_PROJECT_LOCATION,
                              genai_credential_path=GOOGLE_APPLICATION_CREDENTIALS)

response = patient_agent(
    user_prompt="How can I help you?",
)

print(response)

# Doctor Agent
doctor_agent = DoctorAgent('gpt-4o')
print(doctor_agent.system_prompt)


# Example response:
# > I'm experiencing some concerning symptoms, but I can't recall any specific medical history.
# > You are playing the role of a kind and patient doctor...
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