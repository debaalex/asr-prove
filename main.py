import io
import librosa
from time import time
import numpy as np
import IPython.display as ipd
import grpc
import requests

# NLP proto
import riva_api.riva_nlp_pb2 as rnlp
import riva_api.riva_nlp_pb2_grpc as rnlp_srv

# ASR proto
import riva_api.riva_asr_pb2 as rasr
import riva_api.riva_asr_pb2_grpc as rasr_srv

# TTS proto
import riva_api.riva_tts_pb2 as rtts
import riva_api.riva_tts_pb2_grpc as rtts_srv
import riva_api.riva_audio_pb2 as ra


channel = grpc.insecure_channel('localhost:50051')

riva_asr = rasr_srv.RivaSpeechRecognitionStub(channel)
riva_nlp = rnlp_srv.RivaLanguageUnderstandingStub(channel)
riva_tts = rtts_srv.RivaSpeechSynthesisStub(channel)

path = "/Users/alexdebertolis/PycharmProjects/pythonProject2/Record(online-audio-converter.com).wav" #percorso del file audio 
audio, sr = librosa.core.load(path, sr=None)
with io.open(path, 'rb') as fh:
    content = fh.read()
ipd.Audio(path)

req = rasr.RecognizeRequest()
req.audio = content                                   # raw bytes
req.config.encoding = ra.AudioEncoding.LINEAR_PCM     # Supports LINEAR_PCM, FLAC, MULAW and ALAW audio encodings
req.config.sample_rate_hertz = sr                     # Audio will be resampled if necessary
req.config.language_code = "en-US"                    # Ignored, will route to correct model in future release
req.config.max_alternatives = 1                       # How many top-N hypotheses to return
req.config.enable_automatic_punctuation = True        # Add punctuation when end of VAD detected
req.config.audio_channel_count = 1                    # Mono channel

response = riva_asr.Recognize(req)
asr_best_transcript = response.results[0].alternatives[0].transcript
print("ASR Transcript:", asr_best_transcript)

print("\n\nFull Response Message:")
print(response)
