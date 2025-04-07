import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService, Session

load_dotenv()
API_KEY = os.getenv("IBM_API_KEY")

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token=API_KEY
)
job = service.job('czsx78dqnmvg008v5ny0')
job_result = job.result()

counts = job_result[0].data.meas.get_counts() 
# print(job_result[0].data.meas.get_counts())

# transform this counts into probabilities
probabilities = {key: value / sum(counts.values()) for key, value in counts.items()}
print(probabilities)