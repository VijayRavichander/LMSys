import torch
import torch.nn as nn
from transformers import Gemma2ForSequenceClassification, Gemma2Config, EvalPrediction

class CustomGemma2ForSequenceClassification(Gemma2ForSequenceClassification):
    def __init__(self, config, num_labels_head1=58, num_labels_head2=58):
        super().__init__(config)
        self.num_labels_head1 = num_labels_head1
        self.num_labels_head2 = num_labels_head2
        self.classifier_head1 = nn.Linear(config.hidden_size, num_labels_head1, bias=False)
        self.classifier_head2 = nn.Linear(config.hidden_size, num_labels_head2, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        device = input_ids.device

        if labels is not None:
            labels = labels.to(device)
            outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels[:, 0], output_hidden_states=True)
        else:
            outputs = super().forward(input_ids, attention_mask=attention_mask)
        
        # Idx of Last Token Before Padding 
        last_token_indices = (torch.sum(attention_mask, dim=1) - 1).to(device)
        
        # Embedding of Last Token Before Padding
        last_token_outputs = outputs.hidden_states[-1].to(device)[
            torch.arange(outputs.hidden_states[-1].shape[0], device=device), last_token_indices]
        
        outputs_head1 = self.classifier_head1(last_token_outputs).to(device)
        outputs_head2 = self.classifier_head2(last_token_outputs).to(device)

        if labels is not None:
            labels_head1 = labels[:, 1].to(device)
            labels_head2 = labels[:, 2].to(device)
            
            loss_head1 = nn.CrossEntropyLoss()(outputs_head1, labels_head1)
            loss_head2 = nn.CrossEntropyLoss()(outputs_head2, labels_head2)
            loss = outputs.loss.to(device) + 0.1 * loss_head1 + 0.1 * loss_head2
            
            return {"loss": loss, "logits": (outputs.logits, outputs_head1, outputs_head2)}
        
        else:
            return {"logits": (outputs.logits, outputs_head1, outputs_head2)}

