import json
import boto3


def get_current_sagemaker_instance_name() -> str:
    metadata_file = '/opt/ml/metadata/resource-metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        return metadata['ResourceName']


def shut_down_current_sagemaker_instance() -> None:
    instance_name = get_current_sagemaker_instance_name()
    
    sagemaker = boto3.client('sagemaker')
    sagemaker.stop_notebook_instance(NotebookInstanceName=instance_name)