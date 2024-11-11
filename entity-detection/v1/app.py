from flask import Flask, request, jsonify
import re
from groq import Groq
from dotenv import load_dotenv
import os
from word2number import w2n

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)


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

    # Ensure you provide correct instruction format in the Groq model
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"Extract the following entities from this sentence:\n\n{text}\n\nCustomerName, Price, ItemName, ItemQuantity. Return only the values without any comments.",
        }],
        model="llama3-8b-8192",
    )

    response = chat_completion.choices[0].message.content

    customer_name_match = re.search(r'CustomerName:\s*(.*)', response, re.IGNORECASE)
    price_match = re.search(r'Price:\s*(.*)', response, re.IGNORECASE)
    item_name_match = re.search(r'ItemName:\s*(.*)', response, re.IGNORECASE)
    item_quantity_match = re.search(r'ItemQuantity:\s*(.*)', response, re.IGNORECASE)

    CustomerName = extract_value(customer_name_match)
    Price = extract_value(price_match)
    ItemName = extract_value(item_name_match)
    ItemQuantity = extract_value(item_quantity_match)

    Price = clean_price(Price)
    ItemQuantity = convert_word_to_number(ItemQuantity)

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
