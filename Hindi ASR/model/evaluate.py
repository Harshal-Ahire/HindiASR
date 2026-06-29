from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from evaluate import load
import torch

def evaluate_model(model_path: str, test_split: str = "test"):
    """
    Evaluate fine-tuned model on held-out test set.
    Computes WER and CER separately to diagnose error types.
    """
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    dataset = load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "hi",
        split=test_split
    )

    wer_metric = load("wer")
    cer_metric = load("cer")
    predictions, references = [], []



    with torch.no_grad():
        for sample in dataset:
            input_features = processor(
                sample["audio"]["array"],
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            predictions.append(transcription)
            references.append(sample["sentence"])

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print(f"WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
    return {"wer": wer, "cer": cer}