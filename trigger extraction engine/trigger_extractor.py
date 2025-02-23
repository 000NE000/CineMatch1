import pandas as pd
import os
import json
import asyncio
import aiohttp
import logging
from google import genai
from config.config import GEMINI_CONFIG
from prompt_trigger_extraction import prompt

# Configuration
valuenet_balanced_path = "../data/input/VALUENET_balanced"
output_path = "../data/output/trigger_extraction"
model_name = "gemini-2.0-flash"
max_retries = 3

# Initialize GEMINI client
client = genai.Client(api_key=GEMINI_CONFIG["api_key"])

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("trigger_extraction.log"), logging.StreamHandler()])


class TriggerExtractor:
    def __init__(self, value_name):
        self.value_name = value_name
        self.results = []

    def set_df(self, file_name):
        df = pd.read_csv(os.path.join(valuenet_balanced_path, file_name))

        print("Column names:", df.columns)
        if 'uid' not in df.columns:
            raise ValueError("CSV file must contain 'uid' column.")

        return df

    def create_prompt(self, df):
        prompts = []
        data = {"scenario": df['scenario'].to_list(),
                "label": df['label'].to_list(),
                "value": self.value_name}

        for i in range(len(data["scenario"])):
            prompts.append(prompt.format(
                scenario=data["scenario"][i],
                inducted_value=self.value_name,
                label=data["label"][i]
            ))
        return prompts

    async def fetch_response(self, session, prompt_text, uid):
        """
        Asynchronous function to fetch model response.
        Retries on failure for better reliability.
        """
        for attempt in range(max_retries):
            try:
                # Await the response and capture it in a variable.
                response_obj = await client.models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                )
                # Try to convert the response object to a dictionary.
                # This assumes the object has a to_dict() method or you can use its __dict__.
                if hasattr(response_obj, "to_dict"):
                    response_data = response_obj.to_dict()
                else:
                    response_data = response_obj.__dict__
                # Convert the dictionary to a JSON string.
                response_text = json.dumps(response_data)
                return response_text
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
        async with aiohttp.ClientSession() as session:
            tasks = []

            for index, p in enumerate(prompts):
                uid = df.loc[index, "uid"]
                tasks.append(self.fetch_response(session, p, uid))

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
        output_json = os.path.join(output_path, f"{self.value_name.lower()}_triggers.json")
        output_csv = os.path.join(output_path, f"{self.value_name.lower()}_triggers.csv")

        with open(output_json, "w") as json_file:
            json.dump(self.results, json_file, indent=4)

        pd.DataFrame(self.results).to_csv(output_csv, index=False)
        logging.info("Results saved successfully!")

    def run(self):
        os.makedirs(output_path, exist_ok=True)
        df = self.set_df(f"{self.value_name}.csv")
        prompts = self.create_prompt(df)
        asyncio.run(self.process_prompts(df, prompts))
        self.save_results()


# key = GEMINI_CONFIG["api_key"]
                # async with session.post(
                #         url=f"https://generativelanguage.googleapis.com/v1beta2/models/{model_name}:generateText?key={key}",
                #         json={
                #             "prompt": {
                #                 "text": prompt_text,
                #             },
                #             "maxOutputTokens": 256,
                #             "temperature": 0.2,
                #             "candidateCount": 1,
                #
                #         }
                # )
                #
                #
                # async with client.models.generate_content(
                #         model="gemini-2.0-flash",
                #         contents="Explain how AI works",
                # ) as response:
                #     if response.status == 200:
                #         data = await response.json()
                #         return data[0].get("text", "{}")
                #     else:
                #         logging.warning(
                #             f"Attempt {attempt + 1}: Failed to get response for UID {uid}. Status Code: {response.status}")