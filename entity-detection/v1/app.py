from flask import Flask, request, jsonify
import re
from groq import Groq
from dotenv import load_dotenv
import os
from word2number import w2n  # Importing word2number for conversion

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Groq client with API key
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def extract_value(match):
    """
    Extracts value from regex match, handling cases like 'not mentioned' or empty values.
    """
    if match:
        value = match.group(1).strip()
        if value.lower() in ["not mentioned", "none", ""]:
            return ""
        return value
    return ""

def clean_price(price):
    """
    Cleans price string by removing non-numeric characters except for decimal points.
    """
    return re.sub(r'[^\d.]', '', price)

def convert_word_to_number(word):
    """
    Converts words like 'fifty' into their numerical equivalents.
    """
    try:
        return str(w2n.word_to_num(word))
    except ValueError:
        return word  # Return the word if it can't be converted

def validate_values(price, quantity):
    """
    Validates and ensures the correct relationship between price and quantity values.
    """
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
    """
    Endpoint to extract entities using the Groq API based on the input text.
    """
    data = request.json
    text = data.get('text', '')

    # Make API call to Groq for entity extraction
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"Extract the following entities from this sentence:\n\n{text}\n\nCustomerName, Price, ItemName, ItemQuantity. Return only the values without any comments.",
        }],
        model="llama3-8b-8192",
    )

    response = chat_completion.choices[0].message.content

    # Extract individual values from the response
    customer_name_match = re.search(r'CustomerName:\s*(.*)', response, re.IGNORECASE)
    price_match = re.search(r'Price:\s*(.*)', response, re.IGNORECASE)
    item_name_match = re.search(r'ItemName:\s*(.*)', response, re.IGNORECASE)
    item_quantity_match = re.search(r'ItemQuantity:\s*(.*)', response, re.IGNORECASE)

    CustomerName = extract_value(customer_name_match)
    Price = extract_value(price_match)
    ItemName = extract_value(item_name_match)
    ItemQuantity = extract_value(item_quantity_match)

    # Clean price and convert item quantity from words to numbers
    Price = clean_price(Price)
    ItemQuantity = convert_word_to_number(ItemQuantity)  # Convert item quantity from words to numbers

    # Validate price and quantity values
    Price, ItemQuantity = validate_values(Price, ItemQuantity)

    return jsonify({
        "CustomerName": CustomerName,
        "Price": Price,
        "ItemName": ItemName,
        "ItemQuantity": ItemQuantity
    })

def convert_to_entities(input_text):
    """
    Converts the input text into structured entities.
    """
    # Remove unnecessary leading phrases
    cleaned_text = re.sub(r'^(find all|show|list all|display all)\s+', '', input_text, flags=re.IGNORECASE).strip()

    # Convert words to numbers where applicable
    cleaned_text = re.sub(r'\b(fifty|fifty-five|forty|thirty|twenty|ten|one|two|three|four|five|six|seven|eight|nine|zero)\b',
                          lambda x: convert_word_to_number(x.group(0)), cleaned_text)

    # Pattern for cases like "apples less than 50 rs"
    simple_match = re.match(r'(\w[\w\s]*\w)\s+(less|more)\s+than\s+(\d+)\s*rs', cleaned_text, re.IGNORECASE)

    # Pattern for cases like "apples more than 50 less than 70 rs"
    range_match = re.match(r'(\w[\w\s]*\w)\s+more\s+than\s+(\d+)\s+less\s+than\s+(\d+)\s*rs', cleaned_text, re.IGNORECASE)

    if range_match:
        item = range_match.group(1).strip()
        min_value = range_match.group(2).strip()
        max_value = range_match.group(3).strip()
        return {
            "object": item,
            "action": "range",
            "min": min_value,
            "max": max_value
        }
    elif simple_match:
        item = simple_match.group(1).strip()
        action = simple_match.group(2).strip()
        value = simple_match.group(3).strip()
        return {
            "object": item,
            "action": action,
            "range": value
        }
    else:
        return {
            "error": (
                "Invalid input format. Please provide a sentence in the following format:\n"
                "\"John Doe bought 2 apples for $5\" or\n"
                "\"Find all apples more than 50 rs\"."
            )
        }

@app.route('/extract_entities', methods=['POST'])
def extract_entities_endpoint():
    """
    Endpoint for extracting custom entities from the input text.
    """
    data = request.json
    input_text = data.get('text', '')

    # Call the function to parse the input text into entities
    entities = convert_to_entities(input_text)

    return jsonify(entities)

if __name__ == '__main__':
    app.run(debug=True)
