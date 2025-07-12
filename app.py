from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pdfplumber  # Alternative to PyMuPDF
import requests
import json
import os
from werkzeug.utils import secure_filename
import tempfile
import re

app = Flask(__name__)
CORS(app)

# Configuration
GEMINI_API_KEY = "AIzaSyDlIUJ40aJyMasj22JDR9nFQD9Win0rs68"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Store PDF content in memory (in production, use Redis or database)
pdf_content_store = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def call_gemini_api(prompt, max_tokens=111000):
    """Make API call to Gemini Pro"""
    headers = {
        'Content-Type': 'application/json',
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7,
        }
    }
    
    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                raise Exception("No response from Gemini API")
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        raise Exception("Request timeout - please try again")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and text extraction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text
        text_content = extract_text_from_pdf(filepath)
        
        # Clean up file
        os.remove(filepath)
        
        if not text_content:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Store content with a session ID (in production, use proper session management)
        session_id = hash(text_content) % 1000000
        pdf_content_store[session_id] = text_content
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': filename,
            'text_length': len(text_content)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    """Generate summary using Gemini API"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in pdf_content_store:
            return jsonify({'error': 'PDF content not found. Please upload again.'}), 400
        
        text_content = pdf_content_store[session_id]
        
        # Truncate text if too long (Gemini has input limits)
        if len(text_content) > 888000:
            text_content = text_content[:888000] + "..."
        
        prompt = f"""
        Please provide a comprehensive summary of the following document. 
        Make it well-structured with key points and main ideas and try to use as much simpler words you can:

        {text_content}

        Summary:
        """
        
        summary = call_gemini_api(prompt, max_tokens=111500)
        
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/quizify', methods=['POST'])
def quizify_pdf():
    """Generate quiz questions using Gemini API"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in pdf_content_store:
            return jsonify({'error': 'PDF content not found. Please upload again.'}), 400
        
        text_content = pdf_content_store[session_id]
        
        # Truncate text if too long
        if len(text_content) > 666000:
            text_content = text_content[:666000] + "..."
        
        prompt = f"""
        Based on the following document, create exactly 7 multiple-choice questions with 4 options each.
        Each question should have only one correct answer. Kindly make sure to choose those questions whose length are between 12-25              words and the options should not be in more than 3-4 words
        
        Format your response as a JSON array like this:
        [
          {{
            "question": "Question text here?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Option A"
          }}
        ]
        
        The "answer" field should be the exact text of the correct answer from the options array.
        Make sure questions test understanding of key concepts from the document.
        
        Document:
        {text_content}
        """
        
        response = call_gemini_api(prompt, max_tokens=211000)
        
        # Extract JSON from response
        try:
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                questions = json.loads(json_str)
            else:
                # Fallback parsing
                questions = json.loads(response)
            
            # Validate questions format
            if not isinstance(questions, list) or len(questions) == 0:
                raise ValueError("Invalid questions format")
            
            for q in questions:
                if not all(key in q for key in ['question', 'options', 'answer']):
                    raise ValueError("Missing required fields in question")
                if len(q['options']) != 4:
                    raise ValueError("Each question must have exactly 4 options")
            
            return jsonify({
                'success': True,
                'questions': questions
            })
            
        except (json.JSONDecodeError, ValueError) as e:
            return jsonify({'error': f'Failed to parse quiz questions: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer questions based on PDF content"""
    try:
        data = request.json
        session_id = data.get('session_id')
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        if session_id not in pdf_content_store:
            return jsonify({'error': 'PDF content not found. Please upload again.'}), 400
        
        text_content = pdf_content_store[session_id]
        
        # Truncate text if too long
        if len(text_content) > 777000:
            text_content = text_content[:777000] + "..."
        
        prompt = f"""
        Based on the following document, please answer the question below and try your best to use simple words so that even a class 8th          student can understand and the length of the answer should vary according to user specification and need, if not mentioned that           you try to guess if the answer should be given lengthy or short, but even in the worst scenario, dont use more than 140 words.
        If the answer is not found in the document, please say "I cannot find information about this in the uploaded document."
        
        Document:
        {text_content}
        
        Question: {question}
        
        Answer:
        """
        
        answer = call_gemini_api(prompt, max_tokens=81100)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
