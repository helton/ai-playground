import openai

# Using LM Studio - Local Inference Server (local OpenAI mock server)
# only compatible with openai<=0.28

# base local server url
openai.api_base = "http://localhost:1234/v1"
# no need for an API key
openai.api_key = ""

completion = openai.ChatCompletion.create(
  # model field is currently unused
  model="local-model",
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Introduce yourself."}
  ]
)

print(f"model loaded = {completion['model']}")
print(f"answer = {completion.choices[0].message['content'].strip()}")
