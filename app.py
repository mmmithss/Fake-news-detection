from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load BERT model
model = BertForSequenceClassification.from_pretrained("model")
tokenizer = BertTokenizer.from_pretrained("model")
label_mapping = {
    0: 'False',  # barely-true → False
    1: 'False',  # false → False
    2: 'True',   # half-true → True
    3: 'True',   # mostly-true → True
    4: 'False',  # pants-fire → False
    5: 'True'    # true → True
}
# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
]

def gemini_verification(text):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""Analyze this news statement: "{text}"
    Respond in this exact format:
    Verdict: True/False/Misleading/Unverified
    Confidence: High/Medium/Low
    Explanation: [50-word max explanation]
    Sources: [comma-separated reliable sources]"""
    
    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return parse_gemini_response(response.text)
    except Exception as e:
        return {"error": str(e)}

def parse_gemini_response(response):
    # Always return a consistent structure, even for errors
    default_response = {
        "verdict": "Unverified",
        "confidence": "Low",
        "explanation": "Analysis unavailable",
        "sources": []
    }

    try:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Initialize with defaults
        result = default_response.copy()
        
        for line in lines:
            if line.startswith("Verdict:"):
                result["verdict"] = line.split(": ")[1] if len(line.split(": ")) > 1 else "Unverified"
            elif line.startswith("Confidence:"):
                result["confidence"] = line.split(": ")[1] if len(line.split(": ")) > 1 else "Low"
            elif line.startswith("Explanation:"):
                result["explanation"] = ": ".join(line.split(": ")[1:]) if ": " in line else "No explanation"
            elif line.startswith("Sources:"):
                sources = line.split(": ")[1] if len(line.split(": ")) > 1 else ""
                result["sources"] = [s.strip() for s in sources.split(",")] if sources else []

        return result

    except Exception as e:
        return {**default_response, "error": str(e)}
    


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    if not text.strip():
        return jsonify({"error": "Please enter text to analyze"}), 400
    
    try:
        # BERT Prediction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

        # Group and sum all True-related categories (removed half-true)
        true_categories = {
            'mostly-true': float(probs[3]),
            'true': float(probs[5]),
            'half-true': float(probs[2])
        }
        true_sum = sum(true_categories.values())

        # Group and sum all False-related categories (added half-true)
        false_categories = {
              # Moved here
            'barely-true': float(probs[0]),
            'false': float(probs[1]),
            'pants-fire': float(probs[4])
        }
        false_sum = sum(false_categories.values())

        # Determine final verdict based on aggregated sums
        final_verdict = "True" if true_sum > false_sum else "False"
        final_confidence = true_sum if true_sum > false_sum else false_sum  # Sum-based confidence

        # Gemini Verification
        gemini_result = gemini_verification(text)
        
        return jsonify({
            "bert": {
                "prediction": final_verdict,  # UPDATED: Sum-based verdict
                "confidence": final_confidence,  # UPDATED: Sum-based confidence
                "category_breakdown": {
                    "true": {
                        "categories": true_categories,
                        "total_sum": true_sum
                    },
                    "false": {
                        "categories": false_categories,
                        "total_sum": false_sum
                    }
                },
                "probabilities": {  # Keeping original per-category values
                    "true": float(probs[5]),
                    "mostly-true": float(probs[3]),
                    "half-true": float(probs[2]),  # Now under False
                    "barely-true": float(probs[0]),
                    "false": float(probs[1]),
                    "pants-fire": float(probs[4])
                }
            },
            "gemini": gemini_result,
            "combined_verdict": {
                "final_verdict": final_verdict,
                "verdict_reason": f"True categories sum: {true_sum:.2%}, False categories sum: {false_sum:.2%}",
                "difference": f"{abs(true_sum - false_sum):.2%}"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        
def calculate_final_verdict(bert_pred, gemini):
    verdict_map = {
        "True": "true",
        "False": "pants-fire",
        "Misleading": "barely-true",
        "Unverified": "half-true"
    }
    
    gemini_label = verdict_map.get(gemini.get('verdict', 'Unverified'), 'half-true')
    
    if bert_pred == gemini_label:
        return bert_pred
    if gemini.get('confidence', 'Low') == 'High':
        return gemini_label
    return "half-true"

if __name__ == '__main__':
    app.run(debug=True)