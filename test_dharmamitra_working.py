#!/usr/bin/env python3
"""Working test script for Dharmamitra Tibetan to English translation API"""

import requests
import json
import re

# API configuration for the working endpoint
API_URL = "https://dharmamitra.org/api-search/knn-translate-mitra/"
PASSWORD = "sthiramati"

# Sample Tibetan texts to translate
tibetan_texts = [
    ("བཀྲ་ཤིས་བདེ་ལེགས།", "Tashi Delek - Good luck/blessings"),
    ("སངས་རྒྱས་ལ་སྐྱབས་སུ་མཆི།", "I take refuge in the Buddha"),
    ("ཆོས་ལ་སྐྱབས་སུ་མཆི།", "I take refuge in the Dharma"),
    ("དགེ་འདུན་ལ་སྐྱབས་སུ་མཆི།", "I take refuge in the Sangha"),
    ("སེམས་ཅན་ཐམས་ཅད་བདེ་བ་དང་བདེ་བའི་རྒྱུ་དང་ལྡན་པར་གྱུར་ཅིག", "May all sentient beings have happiness and its causes"),
    ("ཨོཾ་མ་ཎི་པདྨེ་ཧཱུཾ།", "Om Mani Padme Hum - famous mantra"),
]

def parse_sse_response(response_text):
    """Parse Server-Sent Events response to extract the translation."""
    translation = ""
    lines = response_text.strip().split('\n')
    
    for line in lines:
        if line.startswith('data: '):
            # Extract the data part and clean it
            data = line[6:]  # Remove 'data: ' prefix
            # Remove quotes if present
            data = data.strip("'\"")
            translation += data
    
    return translation.strip()

def translate_tibetan(text):
    """Translate Tibetan text using the Dharmamitra API."""
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": text,
        "language": "english",
        "password": PASSWORD,
        "do_grammar": True  # Optional parameter for better grammar
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            # Parse the SSE response
            translation = parse_sse_response(response.text)
            return True, translation
        else:
            return False, f"Error {response.status_code}: {response.text}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Request Error: {e}"
    except Exception as e:
        return False, f"Unexpected Error: {e}"

def main():
    """Main function to test translations."""
    print("Dharmamitra Tibetan to English Translation API Test")
    print(f"Endpoint: {API_URL}")
    print(f"Authentication: Using password in request body")
    print("="*70)
    
    for tibetan, description in tibetan_texts:
        print(f"\nTibetan: {tibetan}")
        print(f"Description: {description}")
        
        success, result = translate_tibetan(tibetan)
        
        if success:
            print(f"Translation: {result}")
        else:
            print(f"Failed: {result}")
        
        print("-"*50)
    
    # Interactive mode
    print("\n" + "="*70)
    print("You can now test with your own Tibetan text.")
    print("Enter 'quit' to exit.")
    print("="*70)
    
    while True:
        user_input = input("\nEnter Tibetan text (or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        if user_input:
            success, result = translate_tibetan(user_input)
            if success:
                print(f"Translation: {result}")
            else:
                print(f"Failed: {result}")

if __name__ == "__main__":
    main()
