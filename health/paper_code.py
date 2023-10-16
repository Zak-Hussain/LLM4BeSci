import torch
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer
)
import evaluate

# For reproducibility
torch.manual_seed(42)

# Initialising model and moving to the GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=1
)
model = model.to(device)

# Splitting the data into train and test sets
dat = dat.train_test_split(test_size=.2, seed=42)

# Setting up training arguments for the trainer
model_name = f"{model_ckpt}-finetuned-health"
batch_size = 8
training_args = TrainingArguments(
    output_dir=model_name,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_strategy="epoch", # log training metrics at every epoch
    evaluation_strategy="epoch",
    num_train_epochs=10,
)


def compute_metrics(eval_preds):
    """Computes the coefficient of determination (R2) on the test set."""
    metric = evaluate.load("r_squared")
    preds, labels = eval_preds
    return {
        "r_squared": metric.compute(predictions=preds, references=labels)
    }


# Fine-tuning and evaluating the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dat['train'],
    eval_dataset=dat['test'],
    compute_metrics=compute_metrics,
)
trainer.train()