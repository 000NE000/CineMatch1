import pandas as pd
import os
import json
import asyncio
import logging
import openai
from config.config import GPT_CONFIG
from prompt_trigger_extraction import prompt

# Configuration
valuenet_balanced_path = "../data/input/VALUENET_balanced"
output_path = "../data/output/trigger_extraction"
model_name = "gpt-4o-mini"
max_retries = 3

# Initialize OpenAI API key
openai.api_key = GPT_CONFIG["api_key"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("trigger_extraction.log"), logging.StreamHandler()]
)


class TriggerExtractor:
    def __init__(self, value_name):
        self.value_name = value_name
        self.results = []

    def set_df(self, file_name):
        df = pd.read_csv(os.path.join(valuenet_balanced_path, file_name))
        logging.info("Column names: %s", df.columns.tolist())
        if 'uid' not in df.columns:
            raise ValueError("CSV file must contain 'uid' column.")
        return df

    def create_prompt(self, df):
        prompts = []
        data = {
            "scenario": df['scenario'].to_list(),
            "label": df['label'].to_list(),
            "value": self.value_name
        }
        for i in range(len(data["scenario"])):
            prompts.append(prompt.format(
                scenario=data["scenario"][i],
                inducted_value=self.value_name,
                label=data["label"][i]
            ))
        return prompts

    async def fetch_response(self, prompt_text, uid):
        """
        Asynchronous function to fetch model response using OpenAI ChatCompletion API.
        Retries on failure for better reliability.
        """
        for attempt in range(max_retries):
            try:
                response = await openai.ChatCompletion.acreate(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=256,
                    temperature=0.2,
                )
                content = response['choices'][0]['message']['content']
                return content
            except Exception as e:
                logging.error(f"Error fetching response for UID {uid} on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2 ** attempt)
        logging.error(f"Max retries reached for UID {uid}")
        return None

    def extract_triggers(self, response_text):
        """
        Extract triggers and explanation from the model response text.
        """
        try:
            response_data = json.loads(response_text)
            triggers = response_data.get("Trigger(s)", ["No relevant event"])
            explanation = response_data.get("Explanation", "")
            return triggers, explanation
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON: {response_text}")
            return ["No relevant event"], ""

    async def process_prompts(self, df, prompts):
        tasks = []
        for index, p in enumerate(prompts):
            uid = df.loc[index, "uid"]
            tasks.append(self.fetch_response(p, uid))
        responses = await asyncio.gather(*tasks)
        for index, response_text in enumerate(responses):
            uid = df.loc[index, "uid"]
            if response_text:
                triggers, explanation = self.extract_triggers(response_text)
                result = {
                    "uid": uid,
                    "value": self.value_name,
                    "label": df.loc[index, "label"],
                    "triggers": triggers,
                    "explanation": explanation,
                    "scenario": df.loc[index, "scenario"]
                }
                self.results.append(result)
                logging.info(f"Processed UID {uid}: {result}")
            else:
                logging.warning(f"No response for UID {uid}")

    def save_results(self):
        os.makedirs(output_path, exist_ok=True)
        output_json = os.path.join(output_path, f"{self.value_name.lower()}_triggers.json")
        output_csv = os.path.join(output_path, f"{self.value_name.lower()}_triggers.csv")
        with open(output_json, "w") as json_file:
            json.dump(self.results, json_file, indent=4)
        pd.DataFrame(self.results).to_csv(output_csv, index=False)
        logging.info("Results saved successfully!")

    def run(self):
        df = self.set_df(f"{self.value_name}.csv")
        prompts = self.create_prompt(df)
        asyncio.run(self.process_prompts(df, prompts))
        self.save_results()


if __name__ == "__main__":
    # Change value_name to the required value (e.g., "ACHIEVEMENT", "POWER", etc.)
    value_name = "ACHIEVEMENT"
    extractor = TriggerExtractor(value_name)
    extractor.run()