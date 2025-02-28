import json

# 새 데이터가 담긴 JSON array 파일 읽기
with open('../../data/output/trigger_extraction/scenario_trigger_annotation.json', 'r') as f:
    new_data = json.load(f)

# 기존 JSONL 파일 읽기 (각 줄을 JSON 객체로 변환)
existing_entries = []
with open('../scenario_bio_processed.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            existing_entries.append(json.loads(line))

# 두 파일의 데이터 수가 동일한지 확인
if len(new_data) != len(existing_entries):
    print("경고: 두 파일의 항목 수가 다릅니다!")
else:
    # 인덱스별로 기존 JSON 객체에 value와 label 추가
    for i in range(len(new_data)):
        # 새로운 데이터에서 value와 label 가져오기
        existing_entries[i]['value'] = new_data[i]['value']
        existing_entries[i]['label'] = new_data[i]['label']

    # 업데이트된 데이터를 JSONL 파일로 다시 저장 (덮어쓰기)
    with open('../scenario_bio_processed.jsonl', 'w') as f:
        for entry in existing_entries:
            f.write(json.dumps(entry) + '\n')