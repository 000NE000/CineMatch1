import json
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer

BIO_TAG_MAP = {
    "O": 0,
    "B-TRIGGER": 1,
    "I-TRIGGER": 2
}


class ScenarioDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        texts: list of str (각 문장이 하나의 예시)
        labels: list of list of int (각 문장의 토큰별 BIO 레이블, 길이는 max_length와 같다고 가정)
        tokenizer: RoBERTa 토크나이저
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        label_ids = [BIO_TAG_MAP.get(tag, 0) for tag in label]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()  # (max_length)
        attention_mask = encoding['attention_mask'].squeeze()  # (max_length)

        if len(label_ids) < self.max_length:  # label_ids 길이가 max_length와 다를 경우 패딩 또는 자르기 (기본 O=0으로 패딩)
            label_ids = label_ids + [0] * (self.max_length - len(label_ids))
        else:
            label_ids = label_ids[:self.max_length]
        label_tensor = torch.tensor(label_ids, dtype=torch.long)
        return input_ids, attention_mask, label_tensor


def load_scenario_data(file_path="../scenario_bio_processed.jsonl"):
    """
    파일에서 JSONL 형식으로 저장된 데이터를 읽어와 필요한 필드를 추출
    필요한 필드:
      - "text": 원본 문장
      - "bio_tags": 토큰별 BIO 레이블 (예: ["B-TRIGGER", "O", ...])
      - "value": 해당 시나리오가 유도하는 가치 (추후 멀티태스크 학습에 활용 가능)
    """
    texts = []
    bio_tags = []
    values = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            bio_tags.append(record["bio_tags"])
            values.append(record["value"])  # 필요 시 멀티태스크 학습에 사용
    print(f"Total examples: {len(texts)}")
    return texts, bio_tags, values


def create_dataloaders(file_path, tokenizer, max_length=128, batch_size=16, split_ratio=(0.8, 0.1, 0.1)):
    # 파일에서 데이터 로드
    texts, bio_tags, labels = load_scenario_data(file_path)

    # ScenarioDataset 생성
    dataset = ScenarioDataset(texts, labels, tokenizer, max_length)
    total = len(dataset)

    # split_ratio에 따라 train/val/test 사이즈 계산 (80/10/10)
    train_size = int(split_ratio[0] * total)
    val_size = int(split_ratio[1] * total)
    test_size = total - train_size - val_size

    # random_split 사용하여 데이터 분할
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    return train_loader, val_loader, test_loader

