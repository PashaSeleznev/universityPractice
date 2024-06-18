import vosk
import pyttsx3
import pyaudio
import json
import replicate
import os

tts = pyttsx3.init()
voices = tts.getProperty('voices')
tts.setProperty('voices', 'ru')

for voice in voices:
    if voice.name == 'Microsoft Irina Desktop - Russian':
        tts.setProperty('voices', voice.id)
        print(voice.id)

model = vosk.Model('model_small')

replicate_api_token = 'r8_7caJbRHMJ7hwDa2mZ5XDqbufOnPai9M3sm1Fb'
os.environ['REPLICATE_API_TOKEN'] = replicate_api_token

record = vosk.KaldiRecognizer(model, 10000)
aud = pyaudio.PyAudio()
stream = aud.open(format=pyaudio.paInt16,
                  channels=1,
                  rate=10000,
                  input=True,
                  frames_per_buffer=10000)
stream.start_stream()


def listening():
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if record.AcceptWaveform(data) and len(data) > 0:
            answer = json.loads(record.Result())
            if answer['text']:
                yield answer['text']


def sendToLlama(text):
    result = []
    for event in replicate.stream(
            "meta/meta-llama-3-70b-instruct",
            input={
                "prompt": text,
                "temperature": 0.1
            },
    ):
        result.append(str(event))
    return result

def speaking(say):
    tts.say(say)
    tts.runAndWait()


for text in listening():
    if text == 'пока':
        speaking('Всего хорошего')
        quit()
    else:
        try:
            print(text)
            result = sendToLlama(text)
            resultString = "".join(result)
        except replicate.exceptions.ReplicateError as error:
            if error.status == 402:
                resultString = 'Ваши попытки закончились'
            else:
                # Handle other potential errors
                print(f"Replicate Error: {error}")
                resultString = 'Ошибка'

        finally:
            speaking(resultString)



