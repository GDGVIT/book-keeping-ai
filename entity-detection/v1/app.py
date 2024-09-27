from flask import Flask, request, jsonify
import re
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def extract_value(match):
    if match:
        value = match.group(1).strip()
        if value.lower() in ["not mentioned", "none", ""]:
            return ""
        return value
    return ""

def clean_price(price):
    return re.sub(r'[^\d.]', '', price)

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
def extract_entities():
    data = request.json
    text = data.get('text', '')

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Extract the following entities from this sentence:\n\n{text}\n\nCustomerName, Price, ItemName, ItemQuantity. Return only the values without any comments.",
            }
        ],
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

    Price, ItemQuantity = validate_values(Price, ItemQuantity)

    return jsonify({
        "CustomerName": CustomerName,
        "Price": Price,
        "ItemName": ItemName,
        "ItemQuantity": ItemQuantity
    })
def convert_to_dml(input_text):
    parsed_data = re.match(r'(.*)\s+less than\s+(\d+)\s+rs', input_text, re.IGNORECASE)
    if parsed_data:
        item = parsed_data.group(1).strip()
        price = int(parsed_data.group(2))
        return f"SELECT {item} FROM Products WHERE Price < {price};"
    else:
        return "Invalid input format. Please provide something like 'apples less than 50 rs'."


@app.route('/convert_to_dml', methods=['POST'])
def convert_to_dml_endpoint():
    data = request.json
    input_text = data.get('text', '')

    dml_code = convert_to_dml(input_text)

    return jsonify({
        "input_text": input_text,
        "dml_code": dml_code
    })

if __name__ == '__main__':
    app.run(debug=True)