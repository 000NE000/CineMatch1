prompt = """
[input]
Extract the triggers from the following scenario, paying attention to the specified Value Induction Level:
- Scenario: {scenario}
- Inducted Value: {inducted_value}
- Value Induction Level: {label}

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
   - Express each trigger as a concise, generalized word or phrase.
   
[Output Format]
Return the output strictly in JSON format with the following keys:
- "Trigger(s)": A list of the identified triggers. If no relevant trigger is found, return ["No relevant event"].
- "Explanation": A short text (optional) explaining how these triggers promote, reduce, or are unrelated to the value.

[Example]
Input:
- Scenario: “I love finding people whose dogs have died and laughing at them.”
- Inducted Value: "Hedonism"
- Value Induction Level: 1

Output:
  "Trigger(s)": ["Moral Violation", "Emotional Expression"],
  "Explanation": "The scenario expresses a disregard for moral norms, which may indirectly emphasize a pursuit of pleasure without empathy."
"""