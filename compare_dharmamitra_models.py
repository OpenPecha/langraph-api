#!/usr/bin/env python3
"""Compare the working Dharmamitra translation models"""

import requests
import json
import time

# Test texts with varying complexity
TEST_TEXTS = [
    {
        "tibetan": "བཀྲ་ཤིས་བདེ་ལེགས།",
        "description": "Simple greeting (Tashi Delek)"
    },
    {
        "tibetan": "སངས་རྒྱས་ལ་སྐྱབས་སུ་མཆི། ཆོས་ལ་སྐྱབས་སུ་མཆི། དགེ་འདུན་ལ་སྐྱབས་སུ་མཆི།",
        "description": "Three Refuges"
    },
    {
        "tibetan": "སེམས་ཅན་ཐམས་ཅད་བདེ་བ་དང་བདེ་བའི་རྒྱུ་དང་ལྡན་པར་གྱུར་ཅིག",
        "description": "Aspiration for beings' happiness"
    },
    {
        "tibetan": "ཨོཾ་མ་ཎི་པདྨེ་ཧཱུཾ།",
        "description": "Om Mani Padme Hum mantra"
    },
    {
        "tibetan": "སྟོང་པ་ཉིད་ཀྱི་དོན་ལ་བསམ་པ།",
        "description": "Contemplating the meaning of emptiness"
    }
]

# Working models configuration
WORKING_MODELS = [
    {
        "name": "Mitra Model (Streaming)",
        "short_name": "mitra",
        "url": "https://dharmamitra.org/api-search/knn-translate-mitra/",
        "streaming": True,
        "payload_func": lambda text: {
            "query": text,
            "language": "english",
            "password": "sthiramati",
            "do_grammar": True
        }
    },
    {
        "name": "Gemini Model (No-stream)",
        "short_name": "gemini",
        "url": "https://dharmamitra.org/api-search/knn-translate-gemini-no-stream1/",
        "streaming": False,
        "payload_func": lambda text: {
            "query": text,
            "language": "english",
            "password": "sthiramati"
        }
    }
]

def parse_sse_response(response_text):
    """Parse Server-Sent Events response."""
    translation = ""
    lines = response_text.strip().split('\n')
    
    for line in lines:
        if line.startswith('data: '):
            data = line[6:].strip("'\"")
            translation += data
    
    return translation.strip()

def translate_with_model(text, model_config):
    """Translate text using a specific model."""
    headers = {"Content-Type": "application/json"}
    payload = model_config['payload_func'](text)
    
    try:
        start_time = time.time()
        response = requests.post(
            model_config['url'],
            headers=headers,
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            if model_config['streaming']:
                translation = parse_sse_response(response.text)
            else:
                result = response.json()
                translation = result.get('translation', str(result))
                # Clean up the translation markers
                translation = translation.replace('🔽🔽', '').strip()
            
            return {
                "success": True,
                "translation": translation,
                "time": elapsed
            }
        else:
            return {
                "success": False,
                "error": f"Status {response.status_code}: {response.text}",
                "time": elapsed
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": 0
        }

def main():
    """Compare translations from different models."""
    print("Dharmamitra Translation Models Comparison")
    print("="*80)
    print(f"Testing {len(WORKING_MODELS)} working models with {len(TEST_TEXTS)} test texts\n")
    
    # Collect all results
    results = []
    
    for test in TEST_TEXTS:
        print(f"\n{'='*80}")
        print(f"Text: {test['tibetan']}")
        print(f"Description: {test['description']}")
        print("-"*80)
        
        text_results = {"text": test, "translations": {}}
        
        for model in WORKING_MODELS:
            print(f"\n{model['name']}:")
            result = translate_with_model(test['tibetan'], model)
            
            if result['success']:
                print(f"  Translation: {result['translation']}")
                print(f"  Time: {result['time']:.2f}s")
                text_results['translations'][model['short_name']] = result['translation']
            else:
                print(f"  Error: {result['error']}")
                text_results['translations'][model['short_name']] = None
            
            time.sleep(0.5)  # Be nice to the API
        
        results.append(text_results)
    
    # Summary comparison
    print("\n" + "="*80)
    print("TRANSLATION COMPARISON SUMMARY")
    print("="*80)
    
    for result in results:
        print(f"\n{result['text']['description']}:")
        print(f"Tibetan: {result['text']['tibetan']}")
        
        for model_name, translation in result['translations'].items():
            if translation:
                print(f"  {model_name.upper()}: {translation}")
            else:
                print(f"  {model_name.upper()}: [Failed]")
    
    # Model characteristics
    print("\n" + "="*80)
    print("MODEL CHARACTERISTICS")
    print("="*80)
    print("\nMitra Model:")
    print("- Streaming response (Server-Sent Events)")
    print("- Generally more literal translations")
    print("- Consistent performance")
    
    print("\nGemini Model:")
    print("- Non-streaming JSON response")
    print("- Sometimes includes translation markers (🔽🔽)")
    print("- May provide more contextual translations")

if __name__ == "__main__":
    main()
