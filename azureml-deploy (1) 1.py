# from azure.ai.ml import MLClient
# from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
# from azure.identity import DefaultAzureCredential

# client = MLClient.from_config(DefaultAzureCredential())

# endpoint = ManagedOnlineEndpoint(
#     name="loan-default-rf-endpoint",
#     auth_mode="key"
# )

# client.online_endpoints.begin_create_or_update(endpoint).result()

# deployment = ManagedOnlineDeployment(
#     name="rf-deployment",
#     endpoint_name="loan-default-rf-endpoint",
#     model="LoanDefaultRandomForest@latest",
#     environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
#     scoring_script="azureml-score.py",
#     code_path=".",
#     instance_type="Standard_D4a_v4",
#     instance_count=1
# )

# client.online_deployments.begin_create_or_update(deployment).result()
# endpoint.traffic = {"rf-deployment": 100}
# client.online_endpoints.begin_create_or_update(endpoint).result()



from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace
 
# Load your Azure ML workspace (make sure config.json is present or pass parameters directly)
ws = Workspace.from_config()
 
# Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=ws.subscription_id,
    resource_group_name=ws.resource_group,
    workspace_name=ws.name
)
 
# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="loan-default-rf-endpoint",
    auth_mode="key"
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()