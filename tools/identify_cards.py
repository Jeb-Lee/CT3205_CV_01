"""
identify_cards.py -- a third, very different approach, kept for reference.

Carried over from the author's earlier CT3079 capstone project. Instead of a
trained model, it sends the photo to a hosted vision-language model (GPT-4o
through Azure OpenAI) and asks it to describe each card in plain English. It
is not one of the two pipelines evaluated in the report and contributes to
none of its numbers (see the report's AI Usage Declaration).

Takes a photo of 1-3 flashcards (each a single number, letter, or shape,
printed in color on a black square) and prints a plain-English description
of what's on them.

Usage:
    python tools/identify_cards.py path/to/photo.jpg

Needs `pip install openai python-dotenv` and a .env file (never committed --
it's in .gitignore) defining AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT and,
optionally, AZURE_OPENAI_DEPLOYMENT.
"""

import os
import sys
import base64

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-10-21",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "receipt-vision")

PROMPT = """You are looking at a photo containing between 1 and 3 flashcards.

Each card is a black square showing exactly one of the following, printed in
a color:
  - a single digit (0-9)
  - a single capital letter (A-Z)
  - a simple shape (for example: circle, left arrow, right arrow)

Describe in plain English what is on each card you can see - its color and
its content - and nothing else. If there is only one card, describe just
that one."""


def identify(image_path: str) -> str:
    """Send one image to the vision model and return its plain-text description."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python identify_cards.py <path-to-image>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print(identify(path))
