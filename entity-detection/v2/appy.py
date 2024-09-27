from flask import Flask, request, jsonify
import spacy
import re

app = Flask(__name__)

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def clean_price(price):
    return re.sub(r'[^\d.]', '', price)

@app.route('/extract', methods=['POST'])
def extract_entities():
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
            break  # Stop after finding the first person

    # If CustomerName is empty, extract using regex to capture names before "bought"
    if not CustomerName:
        name_match = re.search(r'([A-Za-z\s\.]+)(?=\s+bought)', text)
        if name_match:
            CustomerName = name_match.group(1).strip()

    # Clean CustomerName by removing common prefixes
    CustomerName = re.sub(r'^(Mr\.|Mrs\.|Ms\.|Dr\.|\b) ', '', CustomerName).strip()

    # Extract other entities
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "CURRENCY"]:
            Price = clean_price(ent.text)
        elif ent.label_ == "QUANTITY":
            ItemQuantity = ent.text
        elif ent.label_ == "PRODUCT":
            ItemName = ent.text

    # Fallback to regex for item quantity and name extraction
    quantity_match = re.search(r'(\d+)\s+([a-zA-Z]+)\s+for', text, re.IGNORECASE)
    if quantity_match:
        ItemQuantity = quantity_match.group(1)
        ItemName = quantity_match.group(2)

    # Update regex for price extraction
    price_match = re.search(r'for\s*(\d+(\.\d{1,2})?)\s*(\w+)?', text)
    if price_match:
        Price = clean_price(price_match.group(1))

    return jsonify({
        "CustomerName": CustomerName,
        "ItemName": ItemName,
        "ItemQuantity": ItemQuantity,
        "Price": Price
    })

if __name__ == '__main__':
    app.run(debug=True)
