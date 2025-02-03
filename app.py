from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Load the Flan-T5 model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define a function to generate a response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=150, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Set up a Gradio interface
interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask your question..."),
    outputs=gr.Textbox(),
    title="Amine's Personal Helper",
    description="Ask me questions about what to include in your resume!"
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
