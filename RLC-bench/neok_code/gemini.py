# TODO(developer): Vertex AI SDK - uncomment below & run
# pip3 install --upgrade --user google-cloud-aiplatform
# gcloud auth application-default login

# import vertexai
# from vertexai.generative_models import GenerativeModel, Part

# # Initialize Vertex AI
# vertexai.init(project=project_id, location=location)
# # Load the model
# multimodal_model = GenerativeModel(model_name="gemini-1.0-pro-vision-001")
# # Query the model
# response = multimodal_model.generate_content(
#     [
#         # Add an example image
#         Part.from_uri(
#             "gs://generativeai-downloads/images/scones.jpg", mime_type="image/jpeg"
#         ),
#         # Add an example query
#         "what is shown in this image?",
#     ]
# )
# print(response)
# return response.text


import os
from openai import OpenAI
import requests
import time
import json
import time

API_SECRET_KEY = "xxxxxx";
BASE_URL = "https://api.zhizengzeng.com/v1/"

# chat with other model，deepseek-chat
def chat_completions4(query):
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    print(resp)