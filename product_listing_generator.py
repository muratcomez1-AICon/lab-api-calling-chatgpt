
import os
import io
import json
import base64
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: API authentication
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from the environment.")

client = OpenAI(api_key=api_key)

# Step 2: Load the product dataset
print("Loading product dataset...")

dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train[:10]"
)

products_df = pd.DataFrame(dataset)

print(f"Dataset loaded: {len(products_df)} products")
print("Dataset columns:", products_df.columns.tolist())

# Step 3: Encode product images
def encode_image_to_base64(image):
    if isinstance(image, dict):
        image = image.get("image")

    if not isinstance(image, Image.Image):
        raise ValueError("Product image is missing or invalid.")

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Step 4: Create the product listing prompt
def create_product_listing_prompt(
    product_name, price, category, additional_info=None
):
    return f"""
You are an expert e-commerce copywriter.

Product name: {product_name}
Price: ${price:.2f}
Category: {category}
Additional information: {additional_info or "Not provided"}

Analyze the supplied product image and return valid JSON with:
1. title: SEO-friendly product title, maximum 60 characters
2. description: detailed product description, 150-200 words
3. features: list of 5-7 features
4. keywords: 10-15 comma-separated SEO keywords

Only describe visual details supported by the image.
Do not invent specifications, materials, or product benefits.
Return JSON only.
"""


# Step 5: Call the ChatGPT API
def generate_listing(row):
    image = row.get("image")

    if image is None:
        raise ValueError("No image found for this product.")

    product_name = str(
        row.get("productDisplayName", "Fashion product")
    )
    category = str(row.get("articleType", "Fashion"))
    additional_info = (
        f"Colour: {row.get('baseColour', 'Unknown')}; "
        f"Department: {row.get('gender', 'Unknown')}"
    )

    # Example price for the lab if the dataset has no price field
    price = float(row.get("price", 49.99))

    image_base64 = encode_image_to_base64(image)

    prompt = create_product_listing_prompt(
        product_name, price, category, additional_info
    )

    response = client.responses.create(
model="gpt-4o-mini",       
 input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:image/jpeg;base64,{image_base64}"
                    )
                }
            ]
        }]
    )

    raw_output = response.output_text.strip()

    # Handle possible Markdown code fences
    if raw_output.startswith("```"):
        raw_output = raw_output.split("\n", 1)[1]
        raw_output = raw_output.rsplit("```", 1)[0].strip()

    listing = json.loads(raw_output)

    return {
        "product_id": str(row.get("id", "")),
        "product_name": product_name,
        "price": price,
        "category": category,
        "listing": listing
    }


# Step 6: Process products and save results
results = []
errors = []

for index in range(min(3, len(dataset))):
    row = dataset[index]

    try:
        result = generate_listing(row)
        results.append(result)
        print(f"Successfully processed product {index + 1}")

    except Exception as error:
        errors.append({
            "product_index": index + 1,
            "error": str(error)
        })
        print(f"Failed product {index + 1}: {error}")

    time.sleep(1)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

with open(
    output_dir / "generated_listings.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2, ensure_ascii=False)

with open(
    output_dir / "errors.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(errors, file, indent=2, ensure_ascii=False)

print(f"Completed: {len(results)} listings generated.")
print(f"Errors: {len(errors)}")
print("Results saved in the output folder.")
