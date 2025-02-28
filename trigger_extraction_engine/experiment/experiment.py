import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer, AdamW
from sklearn.metrics import classification_report
import numpy as np
import random
from trigger_extraction_engine.experiment.ScenarioDataset import create_dataloaders
from trigger_extraction_engine.model.baseline_model import BaselineModel
from trigger_extraction_engine.model.improved_model import ImprovedModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from trigger_extraction_engine.experiment.FocalLoss import FocalLoss



NUM_LABELS = 3  # 0: O, 1: B-TRIGGER, 2: I-TRIGGER

# 파일 경로 및 토크나이저 설정
file_path = "../scenario_bio_processed.jsonl"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# create_dataloaders 함수를 사용해 DataLoader 생성 (80/10/10 분할)
train_loader, val_loader, test_loader = create_dataloaders(
    file_path,
    tokenizer,
    max_length=128,
    batch_size=16,
    split_ratio=(0.8, 0.1, 0.1)
)

# 분할된 데이터셋 크기 출력
print(f"Train size: {len(train_loader.dataset)}")
print(f"Validation size: {len(val_loader.dataset)}")
print(f"Test size: {len(test_loader.dataset)}")



# --- 3. 학습 및 평가 함수 정의 ---

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in dataloader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        if isinstance(model, BaselineModel):
            logits = model(input_ids, attention_mask)
        else:
            outputs = model(input_ids, attention_mask)
            logits = outputs["token_logits"]
        loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            if isinstance(model, BaselineModel):
                logits = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids, attention_mask)
                logits = outputs["token_logits"]
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return total_loss / len(dataloader), all_preds, all_labels


def compute_sentence_level_metrics(preds, labels, b_trigger_label=1):
    """
    preds, labels: 리스트(각 문장은 토큰 시퀀스)
    문장 단위로, 해당 문장 내에 최소 한 개의 B-TRIGGER (label==1)이 있으면 1, 없으면 0으로 변환한 후 accuracy 계산
    """
    sentence_preds = []
    sentence_labels = []
    for pred_seq, label_seq in zip(preds, labels):
        pred_trigger = int(b_trigger_label in pred_seq)
        true_trigger = int(b_trigger_label in label_seq)
        sentence_preds.append(pred_trigger)
        sentence_labels.append(true_trigger)
    return sentence_preds, sentence_labels


# --- 4. 메인 학습 및 평가 루프 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = torch.tensor([1.0, 5.0, 5.0]).to(device)
criterion = FocalLoss(gamma=2.0, weight=class_weights, reduction='mean')

def run_training(model, train_loader, val_loader, num_epochs=3):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
    model.load_state_dict(best_model_state)
    return model


# --- 5. 두 모델 학습 및 평가 ---
print("\nTraining base Model ") #[시발!!!]

baseline_model = BaselineModel(num_labels=NUM_LABELS)
baseline_model = run_training(baseline_model, train_loader, val_loader, num_epochs=3)

print("Training imp Model... (with Hierarchical Attention)...") #[시발!!!]
improved_model = ImprovedModel(num_token_labels=NUM_LABELS) #[시발!!!]
improved_model = run_training(improved_model, train_loader, val_loader, num_epochs=3)

# --- 6. 테스트 데이터 평가 및 결과 비교 ---





print("\nEvaluating Baseline Model on Test Set...")
test_loss_base, preds_base, labels_base = evaluate_model(baseline_model, test_loader, criterion, device)
print(f"Baseline Test Loss: {test_loss_base:.4f}")
print("Classification Report (Baseline - Token Level):")
print(classification_report(np.array(labels_base).flatten(), np.array(preds_base).flatten(), digits=4))

print("\nEvaluating Improved Model on Test Set...")
test_loss_imp, preds_imp, labels_imp = evaluate_model(improved_model, test_loader, criterion, device)
print(f"Improved Test Loss: {test_loss_imp:.4f}")
print("Classification Report (Improved - Token Level):")
print(classification_report(np.array(labels_imp).flatten(), np.array(preds_imp).flatten(), digits=4))

# --- 7. 문장 수준 평가 ---
# 각 문장의 토큰 시퀀스에서 최소한 하나의 B-TRIGGER (label 1)가 존재하는지 여부를 기준으로 문장-level 평가 진행
sent_preds_base, sent_labels = compute_sentence_level_metrics(preds_base, labels_base, b_trigger_label=1)
sent_preds_imp, _ = compute_sentence_level_metrics(preds_imp, labels_imp, b_trigger_label=1)

print("\nSentence-level Trigger Prediction Report (Baseline):")
print(classification_report(sent_labels, sent_preds_base, labels=[0, 1],
                            target_names=["No Trigger", "Trigger"], digits=4))

print("\nSentence-level Trigger Prediction Report (Improved):")
print(classification_report(sent_labels, sent_preds_imp, labels=[0, 1],
                            target_names=["No Trigger", "Trigger"], digits=4))


# --- 8. 결과 해석 ---
"""
결과 해석:
1. Token-level 평가:
   - B-TRIGGER Recall: 개선 모델의 classification_report에서 B-TRIGGER (label 1) 클래스의 Recall이 Baseline 모델에 비해 개선되었는지 확인합니다.
   - O 클래스 Precision: O (label 0) 클래스의 Precision이 개선 모델에서 높아졌는지 확인합니다.
2. Sentence-level 평가:
   - 문장 단위로, 최소한 하나의 B-TRIGGER가 예측된 문장과 실제 트리거가 존재한 문장의 비율을 비교하여, 
     문장 수준의 트리거 예측 정확도가 개선되었는지 확인할 수 있습니다.
3. 전반적으로 개선 모델(계층적 attention 도입)이 문맥적 정보를 보다 잘 반영하여, 
   토큰 및 문장 수준에서 모두 트리거의 시작 위치 예측(Recall)과 불필요한 예측(O 클래스 Precision)에서 성능 향상을 보였음을 기대합니다.

위 결과를 토대로, 최종적으로 B-TRIGGER의 Recall 개선 및 O 클래스의 Precision 향상 여부와 함께, 문장 수준의 트리거 예측 정확도가 개선되었는지 정량적으로 해석할 수 있습니다.
"""