import os
import boto3
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.inputs import TrainingInput

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
custom_image_uri = "123456789012.dkr.ecr.ap-south-1.amazonaws.com/cgpa-custom-image:latest"

def get_pipeline(region, role, default_bucket, **kwargs):
    pipeline_session = PipelineSession()
    
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{default_bucket}/cgpa.csv"
    )

    # Step 1: Preprocessing using ScriptProcessor
    preprocess_processor = ScriptProcessor(
        image_uri=custom_image_uri,
        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        base_job_name="cgpa-preprocess",
        sagemaker_session=pipeline_session
    )

    step_preprocess = ProcessingStep(
        name="PreprocessCGPAData",
        processor=preprocess_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/output"
            )
        ],
        code=os.path.join(BASE_DIR, "src/preprocess.py")
    )

    # Step 2: Training using custom Docker image
    estimator = Estimator(
        image_uri=custom_image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=f"s3://{default_bucket}/model-output",
        base_job_name="cgpa-train",
        sagemaker_session=pipeline_session,
        entry_point="train.py",
        source_dir=os.path.join(BASE_DIR, "src")
    )

    step_train = TrainingStep(
        name="TrainCGPAModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    # Define the pipeline
    pipeline = Pipeline(
        name="CGPA-Prediction-Pipeline",
        parameters=[input_data],
        steps=[step_preprocess, step_train],
        sagemaker_session=pipeline_session
    )

    return pipeline
