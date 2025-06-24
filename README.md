# NEURAL-STYLE-TRANSFER
# TEXT-SUMMARIZATION-TOOL
*COMPANY*: CODETECH IT SOLUTIONS
*NAME*: MAHANTHI PREMKUMAR
*INTERN ID*: CT04DF1723
*DOMAIN*: ARTIFICIAL INTELLIGENCE
*DURATION*: 4 WEEKS
*MENTOR*: NEELA SANTOSH

# THIS TASK NAME IS NEURAL STYLE TRANSFER.IT IS ESPECIALLY USED IN DEEP LEARRING AND COMPUTER VISION.THAT IS USED TO COMBINE THE CONTENT OF ONE IMAGE WITH THE STYLE OF ANOTHER TO CREATE A UNIQE,STYLIZED OUTPUT IMAGE.TOOLS USED ARE TensorFlow/Keras,PyTorch,OpenCV.


# OUTPUT 1

![Screenshot 2025-06-23 015558](https://github.com/user-attachments/assets/689d5b22-f3a8-4c0e-9179-f02b219046d7)

ChatGPT
You said:
TEXT SUMMARIZATION TOOL USING ARTIFICIAL INTELLIGENCE 
ChatGPT said:
Here’s a clear and structured outline for building a Text Summarization Tool using Artificial Intelligence:

📄 Text Summarization Tool Using Artificial Intelligence
✅ Project Overview:
Develop a tool that can automatically summarize long pieces of text into concise and informative summaries using AI-based natural language processing (NLP) techniques.

🎯 Key Objectives:
Input: Accept long text documents or articles.

Output: Generate accurate, meaningful, and grammatically correct summaries.

Support both:

Extractive Summarization: Selecting key sentences directly from the text.

Abstractive Summarization: Generating new sentences that convey the core ideas.

🛠 Tools and Technologies:
Programming Language: Python

Libraries:

NLTK, Spacy (for preprocessing)

Scikit-learn (for extractive methods)

Transformers (for abstractive models using pre-trained models like BART, T5)

Frameworks: Hugging Face Transformers, Flask (for web app)

📂 Project Components:
Text Preprocessing

Tokenization

Stop-word removal

Sentence segmentation

Summarization Techniques

Extractive Summarization: Using algorithms like TextRank.

Abstractive Summarization: Using pre-trained models like:

BART (Bidirectional Auto-Regressive Transformer)

T5 (Text-To-Text Transfer Transformer)

Model Integration

Load pre-trained summarization models.

Accept user input text and return summary.

User Interface (Optional)

Simple web app using Flask or Streamlit to input text and display summaries.

🚀 Sample Python Implementation
📌 Extractive Summarization Example:
from gensim.summarization import summarize

text = """Your long input text here..."""

summary = summarize(text, ratio=0.3)
print("Extractive Summary:")
print(summary)
📌 Abstractive Summarization Example using Hugging Face:
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """Your long input text here..."""

summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
print("Abstractive Summary:")
print(summary[0]['summary_text'])
✅ Deliverables:
Python notebook or script for summarization

Example input-output results

(Optional) Web-based text summarization interface

🔥 Further Improvements:
Add multi-language support.

Deploy as a REST API.

Add summarization for audio or PDF files.

If you want, I can help you write the full Python code, create a notebook, or build a simple web app interface. Would you like that? 😊

🎯 Key Objectives:
Input:

Content Image (the image you want to transform)

Style Image (the artistic style you want to apply)

Output:

A new image that retains the content of the content image and the artistic style of the style image.

🛠 Tools and Technologies:
Programming Language: Python

Libraries:

PyTorch or TensorFlow

torchvision for pre-trained models

Matplotlib and PIL for image processing

📂 Project Components:
Load Images

Resize and preprocess content and style images.

Pre-trained Model

Use VGG-19 (commonly used for feature extraction in NST).

Feature Extraction

Extract content and style features from the images using the VGG-19 network.

Loss Functions

Content Loss: Measures the difference in content between the generated and content image.

Style Loss: Measures the difference in style between the generated and style image (using Gram matrices).

Optimization

Iteratively update the generated image to minimize content and style losses.

✅ Deliverables:
Python notebook with full NST implementation.

Input and output image examples.

(Optional) Flask or Streamlit interface for image upload and real-time style transfer.

🔥 Potential Enhancements:
Real-time NST using feedforward networks.

Deploy NST as a web service.

Support video style transfer.

If you’d like, I can:

🔧 Build the full Python notebook for you.

🌐 Create a simple web app to try different images.

🎥 Help with video-based style transfer.
