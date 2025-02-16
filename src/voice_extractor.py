import os                                                                                                                                                                                                          
import torch
import whisperx
from pydub import AudioSegment
from huggingface_hub import login

class VoiceExtractor:
    def __init__(self, audio_file, save_folder="labeled_voices", device="cuda"):
        self.audio_file = audio_file
        self.save_folder = save_folder
        self.device = device
        self.model = None
        self.result = None
        self.diarize_segments = None
        os.makedirs(save_folder, exist_ok=True)

    def check_cuda(self):
        print("CUDA Available:", torch.cuda.is_available())
        print("CUDA Version:", torch.version.cuda)

    def load_model(self, model_name, compute_type):
        self.model = whisperx.load_model(model_name, self.device, compute_type=compute_type)

    def transcribe_audio(self, batch_size):
        audio = whisperx.load_audio(self.audio_file)
        self.result = self.model.transcribe(audio, batch_size=batch_size)
        print("Segments before alignment:", self.result["segments"])

    #change align model if better is found
    def align_audio(self, align_model="infinitejoy/wav2vec2-large-xls-r-300m-bulgarian"):
        audio = whisperx.load_audio(self.audio_file)
        
        if self.result is None:
            raise Exception("Audio not transcribed!")
            
        l_a, metadata = whisperx.load_align_model(language_code=self.result["language"], device=self.device, model_name=align_model)
        self.result = whisperx.align(self.result["segments"], l_a, metadata, audio, self.device, return_char_alignments=False)
        print("Segments after alignment:", self.result["segments"])

    def authenticate_huggingface(self, hf_api_key):
        if not hf_api_key:
            raise Exception("Huggingface API key missing!")
            
        login(token=hf_api_key)

    def diarize_speakers(self, min_speakers=2, max_speakers=2):
        audio = whisperx.load_audio(self.audio_file)
        diarize_model = whisperx.DiarizationPipeline(device=self.device)
        self.diarize_segments = diarize_model(audio, min_speakers=min_speakers,
                                              max_speakers=max_speakers)
        if not self.diarize_segments:
            raise Exception("Could not diarize segments!")
            
        self.result = whisperx.assign_word_speakers(self.diarize_segments, self.result)
        print("Diarization Segments:", self.diarize_segments)
        print("Segments with speaker IDs:", self.result["segments"])
            
    
    def save_speaker_segments(self, speaker_mapping=None):
        if speaker_mapping is None:
            speaker_mapping = {}
        
        audio = AudioSegment.from_file(self.audio_file)
        for i, segment in enumerate(self.result["segments"]):
            start_time = segment["start"] * 1000
            end_time = segment["end"] * 1000
            speaker_id = segment.get("speaker", "unknown")
            speaker = speaker_mapping.get(speaker_id, speaker_id)
            segment_audio = audio[start_time:end_time]
            segment_filename = f"{self.save_folder}/speaker_{speaker}_segment_{i}.wav"
            segment_audio.export(segment_filename, format="wav")
            print(f"Saved: {segment_filename}")
   
