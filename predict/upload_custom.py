""" 
    Deploy Flan-T5-XXL in Vertex AI using Uvicorn server
    The deployment uses a a2-highgpu-1g machine type
"""
    
from google.cloud import aiplatform

STAGING_BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'

aiplatform.init(project=PROJECT_ID, staging_bucket=STAGING_BUCKET, location=LOCATION)

DEPLOY_IMAGE = 'europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/flan-t5-xxl-uvicorn' 
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORTS = [7080]

model = aiplatform.Model.upload(
    display_name=f'custom-finetuning-flan-t5-xxl',    
    description=f'Finetuned Flan T5 model with Uvicorn and FastAPI',
    serving_container_image_uri=DEPLOY_IMAGE,
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=SERVING_CONTAINER_PORTS,
)
print(model.resource_name)

# Retrieve a Model on Vertex
model = aiplatform.Model(model.resource_name)

# Deploy model
endpoint = model.deploy(
     machine_type='a2-highgpu-1g',
     traffic_split={"0": 100}, 
     min_replica_count=1,
     max_replica_count=1,
     accelerator_type= "NVIDIA_TESLA_A100",    
     accelerator_count=1,
     traffic_percentage=100,
     deploy_request_timeout=1200,
     sync=True,
)
endpoint.wait()