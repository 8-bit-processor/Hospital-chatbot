from similarity_model_and_neural_net import SimilarityModel, NeuralNetwork
from neural_net_text_preprocessor import TextPreprocessor
import os
import nltk
import numpy as np

# Ensure NLTK punkt tokenizer is downloaded
# This part is usually handled by download_punkt.py, but we'll include it here for self-containment
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
except nltk.downloader.download_error:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', download_dir=nltk_data_dir)

def main():
    print("Initializing Chatbot...")

    # Define intents and their corresponding responses/actions
    # The 'question' part should be representative phrases for the intent
    from intents import qa_pairs

    # Initialize the SimilarityModel
    similarity_model = SimilarityModel()
    similarity_model.load_model() # Load the sentence transformer model

    # Build the index with our QA pairs
    similarity_model.build_index(qa_pairs)

    # Prepare data for Neural Network
    unique_intents = sorted(list(set([a for q, a in qa_pairs])))
    intent_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
    idx_to_intent = {i: intent for intent, i in intent_to_idx.items()}

    nn_questions = [q for q, a in qa_pairs]
    nn_labels = [intent_to_idx[a] for q, a in qa_pairs]

    # Generate embeddings for NN training
    print("Generating embeddings for Neural Network training...")
    nn_X = similarity_model.model.encode(nn_questions, convert_to_tensor=False)
    nn_y = np.zeros((len(nn_labels), len(unique_intents)))
    for i, label_idx in enumerate(nn_labels):
        nn_y[i, label_idx] = 1

    # Initialize and train the Neural Network
    input_size = nn_X.shape[1] # Embedding dimension
    output_size = len(unique_intents)
    neural_net = NeuralNetwork(input_size=input_size, output_size=output_size, hidden_layers=[128, 64], learning_rate=0.001)

    print("Training Neural Network...")
    # You might want to adjust training_iterations based on your dataset size and desired accuracy
    neural_net.train(nn_X, nn_y, training_iterations=5000, validation_split=0.1, patience=500, progress_callback=lambda i, tr_err, val_err, acc: print(f"Epoch {i}: Train Error {tr_err:.4f}, Val Error {val_err:.4f}, Accuracy {acc:.4f}") if i % 500 == 0 else None)
    print("Neural Network training complete.")

    # Initialize the TextPreprocessor (we'll use it for basic normalization)
    text_preprocessor = TextPreprocessor(remove_punctuation=True, use_stemming=False)

    print("Chatbot is ready. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # Preprocess user input
        # For sentence-transformers, we primarily need normalization, not vectorization
        normalized_input = text_preprocessor.normalize_sentence(user_input)

        # Find the best match using SimilarityModel
        best_match_answer, score = similarity_model.find_best_match(normalized_input)

        # Neural Network prediction
        nn_input_embedding = similarity_model.model.encode(normalized_input, convert_to_tensor=False).reshape(1, -1)
        nn_prediction = neural_net.predict(nn_input_embedding)
        nn_predicted_idx = np.argmax(nn_prediction)
        nn_predicted_intent = idx_to_intent[nn_predicted_idx]
        nn_confidence = nn_prediction[0, nn_predicted_idx]

        print(f"Similarity Model Guess: {best_match_answer} (Score: {score:.2f})")
        print(f"Neural Network Guess: {nn_predicted_intent} (Confidence: {nn_confidence:.2f})")

        if best_match_answer and score > 0.5: # You can adjust this threshold
            if best_match_answer == "ACTION:SCHEDULE_APPOINTMENT":
                print("Chatbot: I can help you schedule a general appointment. What type of specialty clinic appointment are you looking for?")
            elif best_match_answer == "ACTION:SCHEDULE_NEUROLOGY_APPOINTMENT":
                print("Chatbot: I can help you schedule a neurology appointment. Please provide your preferred date and time.")
            elif best_match_answer == "ACTION:SCHEDULE_ORTHOPEDICS_APPOINTMENT":
                print("Chatbot: I can help you schedule an orthopedics appointment. Please provide your preferred date and time.")
            elif best_match_answer == "ACTION:SCHEDULE_DERMATOLOGY_APPOINTMENT":
                print("Chatbot: I can help you schedule a dermatology appointment. Please provide your preferred date and time.")
            elif best_match_answer == "ACTION:SCHEDULE_PHYSICAL_THERAPY_APPOINTMENT":
                print("Chatbot: I can help you schedule a physical therapy appointment. Please provide your preferred date and time.")
            elif best_match_answer == "ACTION:MEDICAL_IMAGING_REQUEST":
                print("Chatbot: I can assist with medical imaging requests. What type of imaging do you need (e.g., CT scan, Ultrasound, MRI)?")
            elif best_match_answer == "ACTION:MEDICAL_SUPPLIES_REQUEST":
                print("Chatbot: I can help you with medical supplies. What specific supplies are you looking for (e.g., incontinence supplies, durable medical equipment)?")
            elif best_match_answer == "ACTION:RENEW_MEDICATION":
                print("Chatbot: I can assist with medication renewals. Please provide your prescription number or patient ID.")
            elif best_match_answer == "ACTION:MEDICAL_ADVICE_REFERRAL":
                print("Chatbot: I cannot provide medical advice. Please consult with a healthcare professional or visit our emergency department for urgent concerns.")
            else:
                print(f"Chatbot: {best_match_answer}")
        else:
            print("Chatbot: I'm sorry, I don't understand. Can you please rephrase your question or ask something else?")

if __name__ == "__main__":
    main()
