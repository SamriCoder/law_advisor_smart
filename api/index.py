from flask import Flask, request, jsonify, render_template
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import law_rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from law_rag import LegalRAGSystem

load_dotenv()

app = Flask(__name__, template_folder='../templates')

rag_system = None

def get_rag_system():
    global rag_system
    if rag_system is None:
        rag_system = LegalRAGSystem()
        rag_system.initialize()
    return rag_system

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_query = data.get('message', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Message is required'}), 400
        
        rag = get_rag_system()
        
        if not rag.is_legal_query(user_query):
            response = {
                'response': "🚫 Please ask a proper legal question about Indian criminal law.\n💡 Try asking about crimes, procedures, or your legal rights.",
                'type': 'error'
            }
        else:
            result = rag.get_legal_answer(user_query)
            response = {
                'response': result['result'],
                'sources': [
                    {
                        'source': doc.metadata.get('source', 'Legal Document'),
                        'page': doc.metadata.get('page', 'N/A'),
                        'content': doc.page_content[:300] + "..."
                    }
                    for doc in result.get('source_documents', [])[:3]
                ],
                'type': 'success'
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

# This is required for Vercel
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    app.run()