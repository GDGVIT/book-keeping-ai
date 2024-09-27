from flask import Flask, request, jsonify
import spacy
import re
from word2number import w2n

app = Flask(__name__)

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def extract_value(match):
    if match:
        value = match.group(1).strip()
        return value if value.lower() not in ["not mentioned", "none", ""] else ""
    return ""

def clean_price(price):
    return re.sub(r'[^\d.]', '', price)

def convert_word_to_number(word):
    try:
        return str(w2n.word_to_num(word))
    except ValueError:
        return word

def validate_values(price, quantity):
    if re.match(r'^\d+(\.\d{1,2})?$', price) and re.match(r'^\d+$', quantity):
        return price, quantity
    elif re.match(r'^\d+$', price) and re.match(r'^\d+$', quantity):
        price_value = int(price)
        quantity_value = int(quantity)
        if quantity_value > price_value:
            return str(quantity_value), str(price_value)
    return price, quantity

@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    text = data.get('text', '')

    # Process the text with spaCy
    doc = nlp(text)

    # Initialize variables for extraction
    CustomerName = ""
    ItemName = ""
    ItemQuantity = ""
    Price = ""

    # Extract entities using spaCy
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            CustomerName = ent.text
        elif ent.label_ in ["MONEY", "CURRENCY"]:
            Price = clean_price(ent.text)
        elif ent.label_ == "QUANTITY":
            ItemQuantity = ent.text
        elif ent.label_ == "PRODUCT":
            ItemName = ent.text

    # Fallback to regex for additional extraction
    quantity_match = re.search(r'(\d+)\s+([a-zA-Z]+)\s+for', text, re.IGNORECASE)
    if quantity_match:
        ItemQuantity = quantity_match.group(1)
        ItemName = quantity_match.group(2)

    price_match = re.search(r'for\s*(\d+(\.\d{1,2})?)\s*(\w+)?', text)
    if price_match:
        Price = clean_price(price_match.group(1))

    # Convert ItemQuantity from words to numbers if needed
    ItemQuantity = convert_word_to_number(ItemQuantity)

    # Validate and possibly adjust values
    Price, ItemQuantity = validate_values(Price, ItemQuantity)

    if not CustomerName or not ItemName or not ItemQuantity or not Price:
        return jsonify({
            "error": "Invalid input format. Please provide a sentence in the following format:\n\"John Doe bought 2 apples for $5\"."
        }), 400

    return jsonify({
        "CustomerName": CustomerName,
        "Price": Price,
        "ItemName": ItemName,
        "ItemQuantity": ItemQuantity
    })

@app.route('/extract_entities', methods=['POST'])
def extract_entities_endpoint():
    data = request.json
    input_text = data.get('text', '')

    # Normalize input text
    input_text = input_text.lower().strip()

    # Improved regex patterns for various inputs
    range_pattern = r'(\w+)\s+more\s+than\s+(\d+)\s+less\s+than\s+(\d+)'
    less_than_pattern = r'(\w+)?\s*less\s+than\s+(\d+)'
    more_than_pattern = r'(\w+)?\s*more\s+than\s+(\d+)'

    range_match = re.match(range_pattern, input_text)
    less_than_match = re.match(less_than_pattern, input_text)
    more_than_match = re.match(more_than_pattern, input_text)

    if range_match:
        item = range_match.group(1) if range_match.group(1) else '*'
        min_value = range_match.group(2)
        max_value = range_match.group(3)
        return jsonify({
            "object": item,
            "action": "range",
            "min": min_value,
            "max": max_value
        })

    elif less_than_match:
        item = less_than_match.group(1) if less_than_match.group(1) else '*'
        value = less_than_match.group(2)
        return jsonify({
            "object": item,
            "action": "less",
            "range": value
        })

    elif more_than_match:
        item = more_than_match.group(1) if more_than_match.group(1) else '*'
        value = more_than_match.group(2)
        return jsonify({
            "object": item,
            "action": "more",
            "range": value
        })

    return jsonify({
        "error": "Invalid input format. Please provide a sentence in a recognizable format."
    }), 400

if __name__ == '__main__':
    app.run(debug=True)
