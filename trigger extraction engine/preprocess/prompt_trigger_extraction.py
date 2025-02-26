prompt = """
[input]
- Extract the triggers from the following scenario, paying attention to the specified Value and its Induction Level:
- input csv file has columns which are value,scenario,label
- input data: {csv_text}

[Annotation Guidelines]
1. The "Value Induction Level" indicates:
   - 1: the scenario promotes the value.
   - 0: the scenario is unrelated to the value.
   - -1: the scenario reduces or contradicts the value.
2. Definition of a "trigger":
   - A central event or element in the scenario that drives the narrative and influences the specified value.
3. Instructions:
   - Analyze the entire scenario context to identify any triggers.
   - If multiple triggers exist, list each one separately.
   - Express each trigger as a abstract, generalized word or phrase(at most 2 words).
   
[Output Format]
Return the output strictly in JSON format with the following keys:
For each row, produce a JSON object with keys "Trigger(s)", "value", and "label". Then return them all in a list. No extra text!

- "Trigger(s)": A list of the identified triggers. If no relevant trigger is found, return ["No relevant event"].
- "value": The specified value.
- "label": The specified label.
- "basis expression": The basis for the original expression determined as a trigger
"""