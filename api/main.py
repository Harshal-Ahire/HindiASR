from fastapi import FastAPI, UploadFile, File
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa
import time
import io
from schemas import TranscriptionResponse

app = FastAPI(title="Hindi ASR API", version="1.0.0")

MODEL_PATH = "model/whisper-hindi-finetuned"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe Hindi audio clip.
    Returns transcript, confidence score, and inference latency.
    """
    start_time = time.time()

    audio_bytes = await file.read()
    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000
    )

    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    with torch.no_grad():
        outputs = model.generate(
            input_features,
            return_dict_in_generate=True,
            output_scores=True
        )

    transcript = processor.decode(
        outputs.sequences[0],
        skip_special_tokens=True
    )

    confidence = torch.exp(
        torch.stack(outputs.scores, dim=1).max(dim=-1).values.mean()
    ).item()

    latency = round(time.time() - start_time, 3)

    return TranscriptionResponse(
        transcript=transcript,
        confidence=round(confidence, 4),
        latency_seconds=latency
    )