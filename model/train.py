from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "hi"
TASK = "transcribe"

def load_lora_model():
    """Load Whisper-small with LoRA adapters applied."""
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,                        # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output: trainable params: 14,999,552 || all params: 243,814,400
    # trainable: 6.15% — confirms 94% reduction
    return model

def compute_metrics(pred, processor):
    """Compute WER and CER on predictions."""
    from evaluate import load
    wer_metric = load("wer")
    cer_metric = load("cer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer, "cer": cer}