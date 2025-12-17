import boto3
import time

# --- CONFIGURATION (UPDATE THESE) ---
BUCKET_NAME = "airline-project-abhay-2025"
IMAGE_URI = "044707326235.dkr.ecr.us-east-1.amazonaws.com/airline-train:latest"
ROLE_ARN = "arn:aws:iam::044707326235:role/AirlineSageMakerRole" 

# Define where the data is and where the model goes
s3_input_data = f"s3://{BUCKET_NAME}/data/"
s3_output_model = f"s3://{BUCKET_NAME}/models/"

# Connect to SageMaker
sm_client = boto3.client('sagemaker', region_name='us-east-1')

# Create a unique job name based on the current time
job_name = f"airline-churn-job-{int(time.time())}"
print(f"Starting Training Job: {job_name}")

# --- THE TRIGGER ---
response = sm_client.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        'TrainingImage': IMAGE_URI,
        'TrainingInputMode': 'File',
    },
    RoleArn=ROLE_ARN,
    InputDataConfig=[
        {
            'ChannelName': 'train', # This matches args.train in your code
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': s3_input_data, # Points specifically to your data folder
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        }
    ],
    OutputDataConfig={
        'S3OutputPath': s3_output_model
    },
    ResourceConfig={
        'InstanceType': 'ml.m5.xlarge', # FREE TIER ELIGIBLE (For first 2 months)
        'InstanceCount': 1,
        'VolumeSizeInGB': 10
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 3600 # Stop after 1 hour to prevent accidental bills
    }
)

print(f"Job started! Status: {response['TrainingJobArn']}")