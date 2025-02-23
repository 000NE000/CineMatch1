import pandas as pd
import os
import json
import asyncio
import aiohttp
import logging
from google import genai
from config.config import GEMINI_CONFIG
from prompt_trigger_extraction import prompt

from trigger_extractor import TriggerExtractor

if __name__ == "__main__":
    # Change value_name to the required value (e.g., "ACHIEVEMENT", "POWER", etc.)
    value_name = "ACHIEVEMENT"
    extractor = TriggerExtractor(value_name)
    extractor.run()









