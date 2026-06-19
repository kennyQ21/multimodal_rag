import os, base64
from groq import Groq
client = Groq(api_key='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')

with open('data/crops/7704e1592360ba4dd23a54eff381d1f0.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

res = client.chat.completions.create(
    model='llama-3.2-11b-vision-preview',
    messages=[{
        'role': 'user', 
        'content': [
            {'type': 'text', 'text': 'Transcribe all the text in this image. Do not add any extra commentary or formatting, just the raw text.'}, 
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}
        ]
    }]
)
print(res.choices[0].message.content)
