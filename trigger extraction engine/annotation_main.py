from config.config import GPT_CONFIG
import pandas as pd
import openai
from prompt_trigger_extraction import prompt
import json
import os
import re
from itertools import chain


def make_value_column(file):
    df = pd.read_csv(file)
    df['value'] = df['scenario'].str.extract(r'\[(.*?)\]')[0].str.lower()
    df['scenario'] = df['scenario'].str.replace(r'\[.*?\] ', '', regex=True)
    df = df[['value', 'scenario', 'label']]

    df.to_csv("../data/input/VALUENET_balanced/train_handled_2.csv", index=False)
####

def zero_label_deletion(input_csv, output_csv, column_name, value_to_remove):
    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Step 2: Filter out rows where the column has the specific value
    df_filtered = df[df[column_name] != value_to_remove]

    # Step 3: Save the filtered DataFrame back to a new CSV file
    df_filtered.to_csv(output_csv, index=False)

# zero_label_deletion(
#     input_csv="../data/input/VALUENET_balanced/train_handled_2.csv",
#     output_csv="../data/input/VALUENET_balanced/train_handled_3.csv",
#     column_name="label",
#     value_to_remove=0)




def read_csv(file_path, rows_per_prompt):
    """Read CSV file and yield chunks of rows"""
    df = pd.read_csv(file_path)
    for start in range(0, len(df), rows_per_prompt):
        yield df.iloc[start:start+rows_per_prompt]

def format_prompt(data_chunk):
    """Format the prompt with the extracted rows"""
    csv_text = data_chunk.to_string(index=False)
    return prompt_template.format(csv_text=csv_text)

def call_gpt_4o(prompt):
    """Call GPT-4 Turbo with the constructed prompt"""
    chat_completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
            system_message
        ],
        max_tokens=1000,
        temperature=0.5
    )
    return chat_completion.choices[0].message.content



def save_json(data, output_file):
    """Save data to a JSON file"""
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)



# configuration
csv_file_path = "../data/input/VALUENET_balanced/train_handled_3_part.csv"
prompt_template = prompt
rows_per_prompt = 30
annotated_triggers = "../data/output/trigger_extraction/value_triggers.json"
if os.path.exists(annotated_triggers):
    os.remove(annotated_triggers)
openai.api_key = GPT_CONFIG["api_key"]
all_responses = []

system_message = {
    "role": "system",
    "content": (
        "You are an AI that strictly returns JSON..."
    )
}

for data_chunk in read_csv(csv_file_path, rows_per_prompt):
    prompt = format_prompt(data_chunk)
    response = call_gpt_4o(prompt)
    print("\n--- GPT-4 Turbo Response ---\n")
    print(response)
    # Collect the JSON response
    try:
        # Parse response as JSON and append to the list
        json_response = json.loads(response)
        all_responses.append(json_response)
    except json.JSONDecodeError:
        match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
        if match:
            try:
                json_data = json.loads(match.group(1))
                all_responses.append(json_data)
            except json.JSONDecodeError:
                print("Failed to parse the response as JSON. Skipping this chunk.")
        else:
            print("No valid JSON found. Skipping this chunk.")
flattened_list = list(chain.from_iterable(all_responses))
save_json(flattened_list, annotated_triggers)

