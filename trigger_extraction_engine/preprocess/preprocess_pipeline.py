import json
import numpy as np
import torch
from transformers import RobertaTokenizerFast, RobertaModel
from tqdm import tqdm

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.eval()  # 추론 모드

# -----------------------------------------------------------------------------
# (1) 시나리오 데이터 처리 함수 (한 줄 텍스트)
# -----------------------------------------------------------------------------
def process_scenario_line(line, triggers):
    # (필요시) 텍스트 정규화 line = normalize_text(line)
    triggers = [str(trigger) for trigger in triggers]
    encoding = tokenizer(line, return_offsets_mapping=True)
    tokens = encoding['input_ids']
    offsets = encoding['offset_mapping']
    token_map = [(tokenizer.decode([token]), off) for token, off in zip(tokens, offsets)]
    return {
        "text": line,
        "tokens": tokens,
        "token_map": token_map,
        "extracted_triggers": triggers  # GPT4o로 추출된 트리거 리스트
    }

# -----------------------------------------------------------------------------
# (2) 플롯 데이터 처리 함수 (이미 문장별 리스트가 존재한다고 가정)
# -----------------------------------------------------------------------------
def process_plot_sentences(sentences):
    sentence_data = []

    for i, sent in enumerate(sentences):
        # (필요시) 텍스트 정규화: sent = normalize_text(sent)
        encoding = tokenizer(
            sent,
            max_length=512,  # 각 chunk의 최대 길이
            truncation=True,  # 길이를 초과할 경우 자르기
            stride=128,  # 이전 chunk와 겹치는 부분
            return_overflowing_tokens=True,  # Overflow된 부분 반환
            return_offsets_mapping=True  # 각 토큰의 위치 정보
        )

        # Overflow된 부분 처리
        num_chunks = len(encoding['input_ids'])

        for chunk_idx in range(num_chunks):
            tokens = encoding['input_ids'][chunk_idx]
            offsets = encoding['offset_mapping'][chunk_idx]
            token_map = [(tokenizer.decode([token]), off) for token, off in zip(tokens, offsets)]

            # 각 chunk에 sentence_id와 chunk_index 부여
            sentence_data.append({
                "sentence_id": i,  # 문장 ID
                "chunk_index": chunk_idx,  # Chunk 순서
                "sentence": sent,
                "tokens": tokens,
                "token_map": token_map
            })

    return sentence_data
# -----------------------------------------------------------------------------
# (3) n-gram 후보 추출 함수 (1-gram, 2-gram)
# -----------------------------------------------------------------------------
def extract_ngrams(tokens, ngram_range=(1, 2)):
    ngrams = []
    token_texts = [tokenizer.decode([t]).strip() for t in tokens if t != tokenizer.pad_token_id]
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(token_texts) - n + 1):
            ngram = " ".join(token_texts[i:i+n])
            ngrams.append((ngram, i, i+n))
    return ngrams

# -----------------------------------------------------------------------------
# (4) 임베딩 생성 함수
# -----------------------------------------------------------------------------
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰(첫 번째 토큰) 임베딩 사용
    embedding = outputs.last_hidden_state[0, 0, :].numpy()
    return embedding

# -----------------------------------------------------------------------------
# (5) 코사인 유사도 계산 함수
# -----------------------------------------------------------------------------
def cosine_similarity(vec1, vec2):
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1_norm, vec2_norm)

# -----------------------------------------------------------------------------
# (6) 문장에 대해 하나의 트리거 텍스트를 매핑하는 함수
# -----------------------------------------------------------------------------
def map_one_trigger_to_sentence(trigger_text, sentence_data, similarity_threshold=0.8, ngram_range=(1,4)):
    """
    sentence_data: { 'sentence_id', 'sentence', 'tokens', 'token_map' }
    trigger_text: GPT4o로부터 추출된 트리거
    return: n-gram 매핑 결과 dict or None
    """
    trigger_embedding = get_embedding(trigger_text)
    candidate_ngrams = extract_ngrams(sentence_data['tokens'], ngram_range=ngram_range)
    best_candidate = None
    best_score = -1
    for ngram, start_idx, end_idx in candidate_ngrams:
        candidate_embedding = get_embedding(ngram)
        score = cosine_similarity(trigger_embedding, candidate_embedding)
        if score > best_score:
            best_score = score
            best_candidate = {
                "trigger_text": trigger_text,
                "mapped_ngram": ngram,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "similarity": float(score)
            }
    if best_score >= similarity_threshold:
        return best_candidate
    else:
        return None

# -----------------------------------------------------------------------------
# (6) 시나리오 한 줄에 대해 모든 추출 트리거를 매핑하고 BIO 태깅 생성 (다대다 지원)
# -----------------------------------------------------------------------------
def process_scenario_line_bio(line, triggers, similarity_threshold=0.8, ngram_range=(1,4)):
    # 기본 시나리오 처리 (토큰화 등)
    sdata = process_scenario_line(line, triggers)
    matched_triggers = []
    for trigger in sdata["extracted_triggers"]:
        mapping = map_one_trigger_to_sentence(trigger, sdata, similarity_threshold, ngram_range)
        if mapping is not None:
            matched_triggers.append(mapping)
    sdata["matched_triggers"] = matched_triggers

    # BIO 태그 생성: 초기에는 모든 토큰에 "O" 할당
    bio_tags = ["O"] * len(sdata["tokens"])
    # 각 매핑 결과에 대해 해당 span에 BIO 태그 할당
    for mapping in matched_triggers:
        start = mapping["start_idx"]
        end = mapping["end_idx"]
        if end > start:
            bio_tags[start] = "B-TRIGGER"
            for i in range(start + 1, end):
                bio_tags[i] = "I-TRIGGER"
    sdata["bio_tags"] = bio_tags
    return sdata

# 플롯 -> 겹치는 청크 위치 정보 활용해 동일 트리거 중복 제거 -> 불완전한 트리거 병합
def merge_overlapping_triggers(sentence_data, stride=128):
    """
    겹치는 청크에서 동일 트리거를 병합하는 함수
    - sentence_data: 전처리된 문장 데이터 리스트 (청크 포함)
    - stride: 청크 간 겹침 길이
    """
    merged_results = []
    for i in range(len(sentence_data)):
        current_chunk = sentence_data[i]
        next_chunk = sentence_data[i + 1] if i + 1 < len(sentence_data) else None

        # 현재 청크의 매핑된 트리거 목록
        current_triggers = current_chunk.get("mapped_triggers", [])

        # 다음 청크가 존재하는 경우에만 병합 처리
        if next_chunk:
            next_triggers = next_chunk.get("mapped_triggers", [])
            new_triggers = []

            # 겹치는 부분 확인 및 병합
            for cur_trigger in current_triggers:
                for next_trigger in next_triggers:
                    # 동일한 트리거 텍스트일 때만 병합 시도
                    if cur_trigger["trigger_text"] == next_trigger["trigger_text"]:
                        # 현재 청크의 끝 부분과 다음 청크의 시작 부분이 겹치는지 확인
                        if cur_trigger["end_idx"] > (512 - stride) and next_trigger["start_idx"] < stride:
                            # 트리거가 청크 경계에 걸쳐 있으므로 병합
                            merged_trigger = {
                                "trigger_text": cur_trigger["trigger_text"],
                                "mapped_ngram": cur_trigger["mapped_ngram"] + " " + next_trigger["mapped_ngram"],
                                "start_idx": cur_trigger["start_idx"],
                                "end_idx": next_trigger["end_idx"],
                                "similarity": max(cur_trigger["similarity"], next_trigger["similarity"])
                            }
                            new_triggers.append(merged_trigger)
                        else:
                            # 겹치지 않는 경우는 그대로 추가
                            new_triggers.append(cur_trigger)
                    else:
                        # 트리거 텍스트가 다르면 그대로 추가
                        new_triggers.append(cur_trigger)
            # 중복 제거
            new_triggers = {t["trigger_text"]: t for t in new_triggers}.values()
            current_chunk["merged_triggers"] = list(new_triggers)
        else:
            # 다음 청크가 없으면 그대로 추가
            current_chunk["merged_triggers"] = current_triggers

        merged_results.append(current_chunk)

    return merged_results


# -----------------------------------------------------------------------------
# (9) 결과 저장 (JSONL)
# -----------------------------------------------------------------------------
def save_to_jsonl(data, file_path):
    """
    data: list of dict (문장별로 matched_triggers까지 포함)
    file_path: 저장 경로
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")



# =============================================================================
# 메인 실행부
# =============================================================================
def main():
    # =========================================================================
    # 1. 데이터 준비
    # =========================================================================
    # (A) 시나리오 데이터 준비 / 추출 트리거 준비
    # 시나리오 데이터는 한 줄 텍스트 형태로 준비됨

    # scenario_path = '../data/output/trigger_extraction/scenario_trigger_annotation.json'
    # with open(scenario_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    # scenario_lines = [item['scenario'] for item in data]
    # scenario_triggers = [item['Trigger(s)'] for item in data]
    #
    # # (B) 플롯 데이터 준비
    # # 플롯 데이터는 이미 문장 단위로 분리된 리스트 형태로 준비됨
    plot_path = '../data/output/trigger_extraction/essential_plot_data.json'
    with open(plot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    plot_sentences = [item['segmented_sentences'] for item in data]
    # pass
    # =========================================================================
    # 2. 시나리오 데이터 전처리
    # =========================================================================
    #
    # processed_scenarios = []
    # # tqdm을 사용하여 진행 상태를 표시
    # for line, triggers in tqdm(list(zip(scenario_lines, scenario_triggers)), desc="Processing scenarios"):
    #     sdata = process_scenario_line_bio(line, triggers, similarity_threshold=0.8, ngram_range=(1, 4))
    #     processed_scenarios.append(sdata)
    #
    #
    # # 결과 출력
    # for sdata in processed_scenarios:
    #     decoded_tokens = [tokenizer.decode([t]).strip() for t in sdata["tokens"]]
    #     print("Scenario Text:", sdata["text"])
    #     print("Tokens:", decoded_tokens)
    #     print("Extracted Triggers:", sdata["extracted_triggers"])
    #     print("Matched Triggers:", sdata.get("matched_triggers"))
    #     print("BIO Tags:", sdata["bio_tags"])
    #     print("-" * 50)
    #
    # # 결과 저장 (예: JSONL 파일)
    # with open("scenario_bio_processed.jsonl", "w", encoding="utf-8") as f:
    #     for item in processed_scenarios:
    #         f.write(json.dumps(item, ensure_ascii=False) + "\n")
    # print("Saved scenario BIO processed data to 'scenario_bio_processed.jsonl'")

    # =========================================================================
    # 3. 플롯 데이터 전처리
    # =========================================================================
    # 플롯 데이터는 문장 단위로 전처리 수행 (GPT4o 기반 트리거 추출 없이 전처리만)

    plot_results = process_plot_sentences(plot_sentences)
    # JSON 리스트 형식으로 저장 (JSONL 파일 사용)
    merge_overlapping_triggers(plot_results)
    save_to_jsonl(plot_results, file_path="plot_processed_merged.jsonl")
    print("Saved merged plot preprocessed data to 'plot_processed_merged.jsonl'")


if __name__ == "__main__":
    main()