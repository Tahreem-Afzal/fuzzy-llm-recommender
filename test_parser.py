"""
test_parser.py
--------------
Quick standalone test to check what intent_parser.py actually produces
on your machine, bypassing the web app entirely. Run with:

    python test_parser.py
"""

from intent_parser import parse_intent_keywords

text = "cost doesn't matter, but i only want lightweight models that run quickly."

weights, filters = parse_intent_keywords(text)

print("Input text:", text)
print("Weights:", weights)
print("Filters:", filters)