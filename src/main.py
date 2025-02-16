# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:42:35 2025

@author: Mert_Kamber
"""
import os
from dotenv import load_dotenv

from voice_extractor import VoiceExtractor


if __name__ == "main":
    audio_file = "inpt.wav"
    load_dotenv()
    hf_api_key = os.getenv("HF_TOKEN")
    
    extractor = VoiceExtractor(audio_file)
    extractor.check_cuda()
    extractor.load_model(model_name="large-v3",compute_type='float16') 
    #make int8 if gpu is weak / change model if low on memory
    extractor.transcribe_audio(batch_size=16) #make 8 or 4 if gpu is weak
    extractor.align_audio()
    extractor.authenticate_huggingface(hf_api_key)
    extractor.diarize_speakers()
    
    speaker_mapping = {
        "SPEAKER_00": "Cvetanka_Rizova",
        "SPEAKER_01": "Kiril_Petkov",
    }
    
    extractor.save_speaker_segments(speaker_mapping)