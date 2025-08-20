import time

from patientsim.doctor import DoctorAgent
from patientsim.patient import PatientAgent
from patientsim.utils import log, colorstr
from patientsim.utils.common_utils import detect_termination



class EDSimulation:
    def __init__(self, 
                 patient_agent: DoctorAgent,
                 doctor_agent: PatientAgent,
                 max_inferences: int = 15):
        
        # Initialize simulation parameters
        self.patient_agent = patient_agent
        self.doctor_agent = doctor_agent
        self.max_inferences = max_inferences
        self.current_inference = 0  # Current inference index
        self._sanity_check()
    

    def _sanity_check(self):
        if not self.doctor_agent.max_inferences == self.max_inferences:
            log("The maximum number of inferences between the Doctor agent and the ED simulation does not match.", level="warning")
            log(f"The simulation will start with the value ({self.max_inferences}) configured in the ED simulation, \
                and the Doctor agent system prompt will be updated accordingly.", level="warning")
            self.doctor_agent.max_inferences = self.max_inferences
            self.doctor_agent.build_prompt()
        if not self.doctor_agent.current_inference == self.current_inference:
            log("The current inference index between the Doctor agent and the ED simulation does not match.", level="warning")
            log(f"The simulation will start with the value ({self.current_inference}) configured in the ED simulation, \
                and the Doctor agent system prompt will be updated accordingly.", level="warning")
            self.doctor_agent.current_inference = self.current_inference
            self.doctor_agent.build_prompt()


    def reset_env(self):
        self.current_inference = 0
        

    def update_env(self):
        self.current_inference += 1
        self.doctor_agent.update_system_prompt(self.current_inference)


    def simulate(self, verbose: bool = True):
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
            self.update_env()
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
            if detect_termination(doctor_response):
                break

            # Prevent API timeouts
            time.sleep(1.0)
        
        log("Simulation completed.", color=True)
