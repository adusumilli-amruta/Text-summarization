import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Streamlit UI
st.title("Text Summarization Interface")
st.write("Provide a paragraph of text, and the model will summarize it for you!")

# User input for text
user_input = st.text_area("Enter your text here:", height=200)

# Hugging Face model name
model_name = "Sudarshan00/summarize_model_2"  # Replace with your model name

# Load model and tokenizer from Hugging Face
@st.cache_resource  # Cache the model and tokenizer to avoid reloading
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(model_name)

# Summarize the user input
if st.button("Summarize"):
    if user_input.strip():
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generate summary
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=150,  # Adjust maximum length of the summary
            min_length=30,   # Adjust minimum length of the summary
            length_penalty=2.0,
            num_beams=4,  # Beam search for better results
            early_stopping=True
        )
        # Decode and display the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.write("### Summary:")
        st.write(summary)
    else:
        st.write("Please provide some text to summarize.")