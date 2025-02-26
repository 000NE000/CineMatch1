from datasets import load_dataset

# train.jsonl 파일 로드
dataset = load_dataset('json', data_files={'train': 'path/to/train.jsonl'})

# 90%는 Train, 10%는 Validation으로 나눔
train_valid_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_valid_split['train']
valid_dataset = train_valid_split['test']

# 데이터셋 확인
print(train_dataset)
print(valid_dataset)