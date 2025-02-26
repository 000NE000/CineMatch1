import csv
import json

# CSV 파일 경로와 JSON 파일 경로 설정
csv_file = '../../input/VALUENET_balanced/train_handled_3_part.csv'
json_file = 'value_triggers.json'

# CSV 파일 읽기
scenarios = {}
with open(csv_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        value = row['value'].strip()
        scenario = row['scenario'].strip()
        label = row['label'].strip()
        scenarios[value] = scenario

# JSON 파일 읽기
with open(json_file, mode='r', encoding='utf-8') as file:
    json_data = json.load(file)

# CSV 데이터와 JSON 데이터 통합
for item in json_data:
    value = item.get('value')
    if value in scenarios:
        item['scenario'] = scenarios[value]  # 시나리오 텍스트 추가

# 결과 출력
print(json.dumps(json_data, indent=4, ensure_ascii=False))

# 결과를 JSON 파일로 저장 (선택 사항)
output_file = 'scenario_trigger_annotation.json'
with open(output_file, mode='w', encoding='utf-8') as file:
    json.dump(json_data, file, indent=4, ensure_ascii=False)