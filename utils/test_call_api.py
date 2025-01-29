#!/usr/bin/env python3

import boto3
import json
from datetime import datetime
import sys
import tracking_aws


## TODO: This isn't needed if you have already logged in via `aws sso login`
# session = boto3.Session(profile_name='YOUR_PFOILE_NAME')

def invoke_model_with_profile():
    """
    Invoke the Claude model using an application inference profile ARN.
    This ensures the model invocation is tracked and monitored through your profile.
    """
    try:
        # Initialize the Bedrock Runtime client
        # We use bedrock-runtime for model invocation as it handles the actual inference requests
        bedrock_runtime = tracking_aws.new_default_client()

        # Define our inference profile ARN
        # This ARN represents the application inference profile we created to track model usage
        inference_profile_arn = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w"  # Haiku profile

        # Prepare the request body according to Claude's expected format
        # Note: The format must match Claude's requirements while being routed through our inference profile
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": "What advantages does cloud computing offer to businesses?"
                }
            ]
        }

        # Make the API call to invoke the model
        # Note: We use the inference profile ARN as the modelId parameter

        response_body = bedrock_runtime.invoke_model(
            modelId=inference_profile_arn,  # Using inference profile ARN instead of base model ID
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        # Process and display the response
        # The response comes as a streaming body that we need to read and decode
        # response_body = json.loads(response['body'].read().decode())

        print("\nModel Response:")
        print("-" * 80)
        # print(f"Response Status: {response['ResponseMetadata']['HTTPStatusCode']}")

        # Extract and display the model's response
        # The exact structure depends on the Claude model version being used
        if 'content' in response_body:
            print("\nContent:")
            print(response_body['content'][0]['text'])
        else:
            print("\nResponse Body:")
            print(json.dumps(response_body, indent=2))

        print("-" * 80)

        # Display any additional metadata or usage information if available
        if 'usage' in response_body:
            print("\nUsage Information:")
            print(json.dumps(response_body['usage'], indent=2))

    except Exception as e:
        print(f"\nError invoking model: {str(e)}")
        # Print the full error details to help with debugging
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    invoke_model_with_profile()
