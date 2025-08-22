import time

from patientsim.patient import PatientAgent
from patientsim.admin_staff import AdminStaffAgent
from patientsim.utils import log, colorstr
from patientsim.utils.common_utils import detect_op_termination



class OPSimulation:
    def __init__(self, 
                 patient_agent: PatientAgent,
                 admin_staff_agent: AdminStaffAgent,
                 max_inferences: int = 5):
        
        # Initialize simulation parameters
        self.patient_agent = patient_agent
        self.admin_staff_agent = admin_staff_agent
        self.max_inferences = max_inferences
        self.current_inference = 0  # Current inference index
        self._sanity_check()
    

    def _sanity_check(self):
        """
        Verify and synchronize the maximum number of inference rounds 
        between the Administration Staff agent and the OP simulation.

        If the configured values do not match, a warning is logged and 
        the Administration Staff agent's configuration is updated to align with the 
        OP simulation. The system prompt is also rebuilt accordingly.
        """
        if not self.admin_staff_agent.max_inferences == self.max_inferences:
            log("The maximum number of inferences between the Administration Staff agent and the OP simulation does not match.", level="warning")
            log(f"The simulation will start with the value ({self.max_inferences}) configured in the OP simulation, \
                and the Administration Staff agent system prompt will be updated accordingly.", level="warning")
            self.admin_staff_agent.max_inferences = self.max_inferences
            self.admin_staff_agent.build_prompt()


    def simulate(self, verbose: bool = True) -> list[dict]:
        """
        Run a full conversation simulation between the Administration Staff and Patient agents
        in the emergency department setting.

        The simulation alternates turns between the Administration Staff and Patient until
        the maximum number of inference rounds is reached or early termination
        is detected. During the final turn, the Administration Staff agent should assign the department to the patient.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        
        Returns:
            list[dict]: Dialogue history, where each dict contains:
                - "role" (str): "Staff" or "Patient"
                - "content" (str): Text content of the dialogue turn
        """
        if verbose:
            log(f"Patient prompt:\n{self.patient_agent.system_prompt}")
            log(f"Administration staff prompt:\n{self.admin_staff_agent.system_prompt}")

        # Start conversation
        staff_greet = self.admin_staff_agent.staff_greet
        dialog_history = [{"role": "Staff", "content": staff_greet}]
        role = f"{colorstr('blue', 'Staff')}   [0%]"
        log(f"{role:<23}: {staff_greet}")

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

            # Obtain response from staff
            staff_response = self.admin_staff_agent(
                user_prompt=dialog_history[-1]["content"] + "\nThis is the final turn. Now, you must provide your top5 differential diagnosis." \
                    if inference_idx == self.max_inferences - 1 else dialog_history[-1]["content"],
                using_multi_turn=True,
                verbose=verbose
            )
            dialog_history.append({"role": "Staff", "content": staff_response})
            role = f"{colorstr('blue', 'Staff')}   [{progress}%]"
            log(f"{role:<23}: {staff_response}")

            # If early termination is detected, break the loop
            if detect_op_termination(staff_response):
                break

            # Prevent API timeouts
            time.sleep(1.0)
        
        log("Simulation completed.", color=True)

        return dialog_history
