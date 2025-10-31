import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
import re
from collections import Counter
import pickle
import os

# Download NLTK data for text evaluation
import nltk
try:
    nltk.data.find('corpora/words')
except:
    nltk.download('words')
from nltk.corpus import words as nltk_words

# Global variables
english_vocab = set(w.lower() for w in nltk_words.words())
maxlen = 160

# Character mappings - will be loaded from pickle file
indices_char = {}
char_indices = {}
chars = []

def load_character_mappings():
    """Load character mappings from pickle file."""
    global indices_char, char_indices, chars
    try:
        if os.path.exists('indices_char.pkl'):
            with open('indices_char.pkl', 'rb') as f:
                indices_char = pickle.load(f)
            char_indices = {char: idx for idx, char in indices_char.items()}
            chars = list(indices_char.values())
            print(f"Character mappings loaded successfully. Total characters: {len(chars)}")
        else:
            # Fallback to hardcoded mappings if pickle file not found
            print("indices_char.pkl not found. Using fallback character mappings.")
            indices_char = {0: ' ', 1: '!', 2: '"', 3: "'", 4: ',', 5: '.', 6: ':', 7: ';', 8: '?', 9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z', 35: 'a', 36: 'b', 37: 'c', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'i', 44: 'j', 45: 'k', 46: 'l', 47: 'm', 48: 'n', 49: 'o', 50: 'p', 51: 'q', 52: 'r', 53: 's', 54: 't', 55: 'u', 56: 'v', 57: 'w', 58: 'x', 59: 'y', 60: 'z'}
            char_indices = {char: idx for idx, char in indices_char.items()}
            chars = list(indices_char.values())
    except Exception as e:
        print(f"Error loading character mappings: {e}")
        # Use fallback mappings
        indices_char = {0: ' ', 1: '!', 2: '"', 3: "'", 4: ',', 5: '.', 6: ':', 7: ';', 8: '?', 9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z', 35: 'a', 36: 'b', 37: 'c', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'i', 44: 'j', 45: 'k', 46: 'l', 47: 'm', 48: 'n', 49: 'o', 50: 'p', 51: 'q', 52: 'r', 53: 's', 54: 't', 55: 'u', 56: 'v', 57: 'w', 58: 'x', 59: 'y', 60: 'z'}
        char_indices = {char: idx for idx, char in indices_char.items()}
        chars = list(indices_char.values())

# Load models (you'll need to upload these to your HuggingFace Space)
model_6_4 = None
model_6_6 = None

def load_models():
    global model_6_4, model_6_6
    try:
        if os.path.exists('p2_model6_4.keras'):
            model_6_4 = keras.models.load_model('p2_model6_4.keras')
            print("Model 6-4 loaded successfully")
        else:
            print("Model 6-4 not found")
            
        if os.path.exists('p2_model6_6.keras'):
            model_6_6 = keras.models.load_model('p2_model6_6.keras')
            print("Model 6-6 loaded successfully")
        else:
            print("Model 6-6 not found")
    except Exception as e:
        print(f"Error loading models: {e}")

def sample(preds, temperature=1.0):
    """Sample from a probability array with temperature."""
    preds = np.asarray(preds).astype("float64")
    if temperature == 0:
        return np.argmax(preds)
    preds = np.log(preds + 1e-8) / temperature  # Add small epsilon to prevent log(0)
    exp_preds = np.exp(preds - np.max(preds))  # Subtract max for numerical stability
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, seed_text, length, temperature=0.7):
    """Generate text using the trained model."""
    if model is None:
        return "Model not loaded"
    
    generated = seed_text
    for _ in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(generated[-maxlen:]):
            if char in char_indices:
                x_pred[0, t, char_indices[char]] = 1.0

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]
        generated += next_char

    return generated

def evaluate_text_diversity(text):
    """Evaluate text diversity metrics."""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    unique_words = set(words)
    unique_chars = set(text)

    repeated_sequences = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])

    return {
        'total_chars': len(text),
        'unique_char_count': len(unique_chars),
        'unique_word_count': len(unique_words),
        'unique_word_ratio': len(unique_words) / len(words) if words else 0,
        'repetition_count': repeated_sequences,
        'repetition_rate': repeated_sequences / len(words) if words else 0
    }

def evaluate_text_accuracy(text):
    """Evaluate text accuracy based on English vocabulary."""
    text = text.lower()
    word_list = re.findall(r'\b\w+\b', text)

    known_words = [word for word in word_list if word in english_vocab]
    unknown_words = [word for word in word_list if word not in english_vocab]

    return {
        'total_words': len(word_list),
        'known_english_words': len(known_words),
        'unknown_words': len(unknown_words),
        'accuracy_ratio': len(known_words) / len(word_list) if word_list else 0
    }

def evaluate_generated_text(text):
    """Comprehensive text evaluation."""
    diversity = evaluate_text_diversity(text)
    accuracy = evaluate_text_accuracy(text)
    return {'diversity': diversity, 'accuracy': accuracy}

def preprocess_input(text_input):
    """Preprocess input text to fit model requirements."""
    if len(text_input) < maxlen:
        padded_text_input = text_input.rjust(maxlen, ' ')
    elif len(text_input) > maxlen:
        padded_text_input = text_input[:maxlen]
    else:
        padded_text_input = text_input
    return padded_text_input

def format_evaluation(eval_results):
    """Format evaluation results for display."""
    div = eval_results['diversity']
    acc = eval_results['accuracy']
    
    return f"""
**Diversity Metrics:**
- Total characters: {div['total_chars']}
- Unique characters: {div['unique_char_count']}
- Unique words: {div['unique_word_count']}
- Unique word ratio: {div['unique_word_ratio']:.3f}
- Repetition count: {div['repetition_count']}
- Repetition rate: {div['repetition_rate']:.3f}

**Accuracy Metrics:**
- Total words: {acc['total_words']}
- Known English words: {acc['known_english_words']}
- Unknown words: {acc['unknown_words']}
- Accuracy ratio: {acc['accuracy_ratio']:.3f}
"""

def generate_with_both_models(input_text, temperature, generation_length):
    """Generate text with both models and return results with evaluations."""
    if not input_text.strip():
        return "Please enter some input text.", "", "No evaluation available.", "No evaluation available."
    
    # Preprocess input
    processed_input = preprocess_input(input_text)
    
    # Generate with Model 6-4
    if model_6_4 is not None:
        generated_6_4 = generate_text(model_6_4, processed_input, generation_length, temperature)
        eval_6_4 = evaluate_generated_text(generated_6_4)
        eval_text_6_4 = format_evaluation(eval_6_4)
    else:
        generated_6_4 = "Model 6-4 not available"
        eval_text_6_4 = "Model 6-4 not loaded"
    
    # Generate with Model 6-6
    if model_6_6 is not None:
        generated_6_6 = generate_text(model_6_6, processed_input, generation_length, temperature)
        eval_6_6 = evaluate_generated_text(generated_6_6)
        eval_text_6_6 = format_evaluation(eval_6_6)
    else:
        generated_6_6 = "Model 6-6 not available"
        eval_text_6_6 = "Model 6-6 not loaded"
    
    return generated_6_4, generated_6_6, eval_text_6_4, eval_text_6_6

# Load models and character mappings on startup
load_character_mappings()
load_models()

# Create Gradio interface
with gr.Blocks(title="Harry Potter Text Generator") as demo:
    gr.Markdown("""
    # 🧙‍♂️ Harry Potter Character-Level Text Generator
    
    This app uses two trained character-level neural networks to generate Harry Potter-style text.
    Enter a seed sentence and the models will generate the texts to continue the 'story'. This app was created
    as part of assignment for Deep Learning for NLP (SDGAI).
    
    **Models:**
    - Model 6-4: Model with the best accuracy scores during training
    - Model 6-6: Model that seemed to generate more readable 'sentences'
    
    **Instructions:**
    1. Enter your seed text (will be padded/truncated to 160 characters)
    2. Adjust temperature (0.1 = conservative, 0.9 = creative)
    3. Set generation length (number of characters to generate)
    4. Click Generate to see results from both models
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Seed Text",
                placeholder="Enter your seed sentence here... (e.g., 'Harry walked down the corridor and saw')",
                lines=3,
                value="He watched a group of students in flowing black robes chatter excitedly as a majestic barn owl, a newly purchased pet, hooted softly from its cage."
            )
            
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Controls randomness: lower = more conservative, higher = more creative"
                )
                
                generation_length = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=400,
                    step=50,
                    label="Generation Length",
                    info="Number of characters to generate"
                )
            
            generate_btn = gr.Button("🎭 Generate Text", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Model 6-4 Output")
            output_6_4 = gr.Textbox(
                label="Generated Text (Model 6-4)",
                lines=8,
                interactive=False
            )
            eval_6_4 = gr.Markdown(label="Evaluation Metrics (Model 6-4)")
        
        with gr.Column():
            gr.Markdown("## Model 6-6 Output")
            output_6_6 = gr.Textbox(
                label="Generated Text (Model 6-6)",
                lines=8,
                interactive=False
            )
            eval_6_6 = gr.Markdown(label="Evaluation Metrics (Model 6-6)")
    
    # Connect the generate button to the function
    generate_btn.click(
        fn=generate_with_both_models,
        inputs=[input_text, temperature, generation_length],
        outputs=[output_6_4, output_6_6, eval_6_4, eval_6_6]
    )
    
    gr.Markdown("""
    ---
    
    **About the Evaluation Metrics:**
    
    **Diversity Metrics:**
    - **Unique word ratio**: Higher values indicate more diverse vocabulary
    - **Repetition rate**: Lower values indicate less repetitive text
    
    **Accuracy Metrics:**
    - **Accuracy ratio**: Proportion of generated words that exist in English dictionary
    
    """)

if __name__ == "__main__":
    demo.launch()