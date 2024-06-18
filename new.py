import vosk
import pyttsx3
import pyaudio
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

tts = pyttsx3.init()
voices = tts.getProperty('voices')
tts.setProperty('voice', 'en')

for voice in voices:
    if voice.name == 'Microsoft David Desktop - English (United States)':
        tts.setProperty('voice', voice.id)

model = vosk.Model('model_small_en')

record = vosk.KaldiRecognizer(model, 16000)
aud = pyaudio.PyAudio()
stream = aud.open(format=pyaudio.paInt16,
                  channels=1,
                  rate=16000,
                  input=True,
                  frames_per_buffer=4000)
stream.start_stream()


def listening():
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if record.AcceptWaveform(data) and len(data) > 0:
            answer = json.loads(record.Result())
            if answer['text']:
                yield answer['text']


def sendToGPT(text):
    encoded_input = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True)
    attention_mask = torch.ones_like(encoded_input)

    output = gpt_model.generate(
        input_ids=encoded_input,
        attention_mask=attention_mask,  # Передаем маску внимания
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


def speaking(say):
    tts.say(say)
    tts.runAndWait()


for text in listening():
    if text == 'thank you':
        speaking('have a nice day')
        quit()
    else:
        print(text)
        answer = sendToGPT(text)
        print(answer)
        speaking(answer)
