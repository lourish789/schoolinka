import os
import time
import re
import asyncio
import nest_asyncio
from datetime import datetime
import sqlite3
from contextlib import contextmanager
from flask import Flask, request, jsonify
import threading
import requests
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import logging
from typing import List, Dict, Optional, Tuple
import sys
import pysqlite3
import gspread
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials
from tqdm import tqdm

# Fix SQLite and apply async patch
sys.modules["sqlite3"] = pysqlite3
nest_asyncio.apply()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ai_coach.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
CONFIG = {
    'PINECONE_API_KEY': os.getenv("PINECONE_API_KEY", "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg"),
    'GOOGLE_API_KEY': os.getenv("GOOGLE_API_KEY", "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw"),
    'GREEN_API_ID': os.getenv("GREEN_API_ID_INSTANCE", "7105328354"),
    'GREEN_API_TOKEN': os.getenv("GREEN_API_TOKEN", "2a33db828fe64c57a32debcca8f065cac2f901d270d04347a5"),
    'SPREADSHEET_ID': "1H8HKj8sIdD8mPEsR8x32qIprEa_jr6UDfdUU50IjkHk",
    'DB_PATH': "ai_coach.db",
    'MAX_HISTORY': 20,
    'INDEX_NAME': 'coach'
}

# Validate keys
if not all([CONFIG['PINECONE_API_KEY'], CONFIG['GOOGLE_API_KEY'], 
            CONFIG['GREEN_API_ID'], CONFIG['GREEN_API_TOKEN']]):
    raise ValueError("Missing required API keys")

# Initialize Google Sheets
try:
    gc = gspread.service_account(filename="credentials.json")
    sh = gc.open_by_key(CONFIG['SPREADSHEET_ID'])

    #credential_json = os.getenv('credentials.json')
    #if not credential_json:
        #raise ValueError("No credentials found in environment variable")

    # Parse the JSON string and create credentials
    #credential_info = json.loads(credential_json)
    #credentials = service_account.Credentials.from_service_account_info(
      #  credential_info,
      #  scopes=['https://www.googleapis.com/auth/spreadsheets']
)
    # Authorize gspread
    #gc = gspread.authorize(credentials)
    
    # Get or create worksheets
    try:
        users_sheet = sh.worksheet1#("Users")
    except:
        users_sheet = sh.add_worksheet(title="Users", rows="1000", cols="10")
        users_sheet.append_row(["Timestamp", "Chat ID", "Name", "Email", "Phone", "Location", "Class", "Status"])
    
    try:
        conversations_sheet = sh.worksheet2#("Conversations")
    except:
        conversations_sheet = sh.add_worksheet(title="Conversations", rows="10000", cols="6")
        conversations_sheet.append_row(["Timestamp", "Chat ID", "User Name", "User Message", "Bot Response", "Intent"])
    
    logger.info("Google Sheets connected successfully")
except Exception as e:
    logger.error(f"Google Sheets initialization error: {e}")
    users_sheet = None
    conversations_sheet = None

#import os
#import json
#import gspread
#from google.oauth2 import service_account

# Get the JSON string from an environment variable




# ... rest of your code to open the sheet and update data


# Initialize AI services
try:
    pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    existing_index_names = [index.name for index in existing_indexes.indexes]
    
    if CONFIG['INDEX_NAME'] not in existing_index_names:
        logger.warning(f"Pinecone index '{CONFIG['INDEX_NAME']}' not found. Please run the vector storage script first.")
        pinecone_index = None
    else:
        pinecone_index = pc.Index(CONFIG['INDEX_NAME'])
        logger.info("Pinecone index connected successfully")
    
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=CONFIG['GOOGLE_API_KEY']
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=CONFIG['GOOGLE_API_KEY'],
        temperature=0.4,
        max_tokens=500
    )
    logger.info("AI services initialized successfully")
except Exception as e:
    logger.error(f"Service initialization error: {e}")
    raise


class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    chat_id TEXT PRIMARY KEY,
                    first_name TEXT,
                    email TEXT,
                    phone_number TEXT,
                    location TEXT,
                    class_taught TEXT,
                    profile_complete BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    message_type TEXT CHECK(message_type IN ('user', 'assistant')),
                    message_content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    intent TEXT,
                    FOREIGN KEY (chat_id) REFERENCES users (chat_id)
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat ON conversations (chat_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
            conn.commit()
            logger.info("Database initialized")
    
    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_user(self, chat_id: str) -> Optional[Dict]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE chat_id = ?', (chat_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def save_user(self, chat_id: str, **kwargs) -> bool:
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT chat_id FROM users WHERE chat_id = ?', (chat_id,))
                exists = cursor.fetchone()
                
                if exists:
                    if kwargs:
                        updates = ', '.join([f'{k} = ?' for k in kwargs.keys()])
                        updates += ', last_interaction = ?, total_messages = total_messages + 1'
                        values = list(kwargs.values()) + [datetime.now(), chat_id]
                        cursor.execute(f'UPDATE users SET {updates} WHERE chat_id = ?', values)
                    else:
                        cursor.execute(
                            'UPDATE users SET last_interaction = ?, total_messages = total_messages + 1 WHERE chat_id = ?',
                            (datetime.now(), chat_id)
                        )
                else:
                    fields = ['chat_id'] + list(kwargs.keys()) + ['total_messages']
                    placeholders = ', '.join(['?' for _ in fields])
                    values = [chat_id] + list(kwargs.values()) + [1]
                    cursor.execute(
                        f'INSERT INTO users ({", ".join(fields)}) VALUES ({placeholders})', 
                        values
                    )
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving user {chat_id}: {e}")
            return False
    
    def save_message(self, chat_id: str, msg_type: str, content: str, intent: str = None):
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (chat_id, message_type, message_content, intent)
                    VALUES (?, ?, ?, ?)
                ''', (chat_id, msg_type, content, intent))
                conn.commit()
                self._cleanup_history(chat_id)
        except Exception as e:
            logger.error(f"Error saving message: {e}")
    
    def get_history(self, chat_id: str, limit: int = None) -> List[Dict]:
        limit = limit or CONFIG['MAX_HISTORY']
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_type, message_content, timestamp, intent
                    FROM conversations 
                    WHERE chat_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (chat_id, limit))
                return [dict(row) for row in reversed(cursor.fetchall())]
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
    
    def _cleanup_history(self, chat_id: str):
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM conversations 
                    WHERE chat_id = ? AND id NOT IN (
                        SELECT id FROM conversations 
                        WHERE chat_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    )
                ''', (chat_id, chat_id, CONFIG['MAX_HISTORY'] * 2))
                conn.commit()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


class AICoach:
    """Core AI logic with RAG"""
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
    
    def get_rag_content(self, query: str) -> Tuple[str, List[str]]:
        """Retrieve relevant content from Pinecone"""
        if not self.pinecone_index:
            return "Knowledge base not available.", []
        
        try:
            query_embed = self.embed_model.embed_query(query)
            results = self.pinecone_index.query(
                vector=query_embed,
                top_k=3,
                include_metadata=True
            )
            
            contents, sources = [], []
            for match in results.get('matches', []):
                if match.get('score', 0) > 0.7:
                    text = match['metadata'].get('text', '')[:300]
                    source = match['metadata'].get('source', 'Knowledge Base')
                    if text:
                        contents.append(text)
                        sources.append(source)
            
            return '\n\n'.join(contents) if contents else "", sources
        except Exception as e:
            logger.error(f"RAG error: {e}")
            return "", []
    
    def generate_response(self, message: str, user_profile: Dict, history: List[Dict] = None) -> Tuple[str, str]:
        """Generate AI response"""
        try:
            intent = self._extract_intent(message)
            rag_content, sources = self.get_rag_content(message)
            
            # Build context from conversation history
            context = ""
            if history:
                recent_context = history[-4:]
                context = "\n".join([
                    f"{'Teacher' if msg['message_type'] == 'user' else 'AI Coach'}: {msg['message_content'][:100]}"
                    for msg in recent_context
                ])
            
            system_prompt = f"""You are AI Coach by Schoolinka, helping Nigerian teachers. Provide direct, practical advice.

Teacher Profile:
- Name: {user_profile.get('first_name', 'Teacher')}
- Teaching: {user_profile.get('class_taught', 'Not specified')}
- Location: {user_profile.get('location', 'Nigeria')}

Guidelines:
- Give SHORT, actionable answers (2-3 sentences max)
- Be warm but professional
- Focus on practical solutions for Nigerian classrooms
- No asterisks, bullet points, or special formatting
- Address the specific question directly

{f'Recent conversation context: {context}' if context else ''}

{f'Relevant knowledge: {rag_content}' if rag_content else ''}

Teacher's question: {message}

Provide a clear, concise response:"""
            
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt}
            ])
            
            # Clean response - remove asterisks and excessive formatting
            clean_response = response.content.strip()
            clean_response = clean_response.replace('*', '').replace('**', '')
            clean_response = re.sub(r'\n{3,}', '\n\n', clean_response)
            
            # Ensure response is concise
            sentences = clean_response.split('. ')
            if len(sentences) > 4:
                clean_response = '. '.join(sentences[:4]) + '.'
            
            return clean_response, intent
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm having trouble right now. Please try again in a moment.", "error"
    
    @staticmethod
    def _extract_intent(message: str) -> str:
        """Extract intent from message"""
        msg = message.lower()
        intents = {
            'teaching_strategy': ['strategy', 'method', 'technique', 'how to teach', 'lesson plan', 'explain', 'introduce'],
            'classroom_management': ['discipline', 'behavior', 'manage', 'control', 'disruptive', 'noise', 'attention'],
            'assessment': ['assess', 'evaluate', 'grade', 'test', 'exam', 'score', 'mark', 'performance'],
            'wellbeing': ['stress', 'tired', 'overwhelmed', 'burnout', 'exhausted', 'frustrated', 'difficult'],
            'curriculum': ['curriculum', 'syllabus', 'topic', 'subject', 'content', 'what to teach']
        }
        
        for intent, keywords in intents.items():
            if any(kw in msg for kw in keywords):
                return intent
        return 'general'


def log_to_sheets(chat_id: str, user_name: str, user_message: str, bot_response: str, intent: str = "general"):
    """Log conversation to Google Sheets"""
    if not conversations_sheet:
        logger.warning("Google Sheets not available")
        return
    
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data_to_add = [
            timestamp, 
            chat_id, 
            user_name,
            user_message[:500],
            bot_response[:500],
            intent
        ]
        conversations_sheet.append_row(data_to_add)
        logger.info(f"Conversation logged to Sheets: {chat_id}")
    except Exception as e:
        logger.error(f"Error logging conversation to Sheets: {e}")


def log_user_to_sheets(chat_id: str, user_data: Dict):
    """Log user registration to Google Sheets"""
    if not users_sheet:
        logger.warning("Google Sheets not available")
        return
    
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if user already exists in sheet
        try:
            cell = users_sheet.find(chat_id)
            if cell:
                logger.info(f"User {chat_id} already in sheet, skipping")
                return
        except:
            pass
        
        data_to_add = [
            timestamp,
            chat_id,
            user_data.get('first_name', ''),
            user_data.get('email', ''),
            user_data.get('phone_number', ''),
            user_data.get('location', ''),
            user_data.get('class_taught', ''),
            'Registered'
        ]
        users_sheet.append_row(data_to_add)
        logger.info(f"User registered to Sheets: {chat_id}")
    except Exception as e:
        logger.error(f"Error logging user to Sheets: {e}")


def parse_registration(text: str) -> Dict:
    """Parse user registration details"""
    details = {}
    lines = text.strip().split('\n')
    
    for line in lines:
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key, value = key.strip().lower(), value.strip()
        
        if 'name' in key and len(value) > 1:
            details['first_name'] = value.title()
        elif 'email' in key and '@' in value:
            details['email'] = value.lower()
        elif 'phone' in key and len(value) >= 7:
            details['phone_number'] = value
        elif 'location' in key and len(value) > 2:
            details['location'] = value.title()
        elif 'class' in key and len(value) > 1:
            details['class_taught'] = value.title()
    
    return details


# Initialize components
db = DatabaseManager(CONFIG['DB_PATH'])
ai_coach = AICoach(llm, embed_model, pinecone_index)


def process_message(chat_id: str, text_message: str) -> str:
    """Main message processing logic - TEXT ONLY"""
    try:
        user = db.get_user(chat_id)
        
        # Handle registration
        if not user or not user.get('profile_complete'):
            if not user:
                db.save_user(chat_id, profile_complete=False)
                return (
                    "Hello! I'm AI Coach by Schoolinka.\n\n"
                    "Please share your details in one message:\n\n"
                    "Name: [Your name]\n"
                    "Email: [Your email]\n"
                    "Phone: [Your number]\n"
                    "Location: [City/State]\n"
                    "Class: [Class you teach]\n\n"
                    "Example:\n"
                    "Name: Amina Bello\n"
                    "Email: amina@email.com\n"
                    "Phone: 08012345678\n"
                    "Location: Lagos\n"
                    "Class: Primary 4"
                )
            
            # Parse registration
            details = parse_registration(text_message)
            
            if len(details) >= 4:
                db.save_user(chat_id, profile_complete=True, **details)
                user = db.get_user(chat_id)
                
                # Log user to sheets
                log_user_to_sheets(chat_id, details)
                
                welcome_msg = (
                    f"Welcome, {details.get('first_name', 'Teacher')}!\n\n"
                    f"I'm here to help with your {details.get('class_taught', 'class')}. "
                    f"You can ask me about teaching strategies, classroom management, "
                    f"assessment methods, or any teaching challenges.\n\n"
                    f"What would you like help with today?"
                )
                
                log_to_sheets(chat_id, details.get('first_name', 'Teacher'), text_message, welcome_msg, "registration")
                return welcome_msg
            else:
                return (
                    "Please provide all 5 details in the format shown:\n\n"
                    "Name: [Your full name]\n"
                    "Email: [Your email]\n"
                    "Phone: [Your number]\n"
                    "Location: [Your city]\n"
                    "Class: [Class you teach]"
                )
        
        # Validate text message
        if not text_message or len(text_message.strip()) < 3:
            return "Please send me a question or message about teaching."
        
        # Update user activity
        db.save_user(chat_id)
        
        # Get conversation history
        history = db.get_history(chat_id, limit=10)
        
        # Extract intent and save user message
        intent = ai_coach._extract_intent(text_message)
        db.save_message(chat_id, 'user', text_message, intent=intent)
        
        # Generate AI response
        ai_response, response_intent = ai_coach.generate_response(text_message, user, history)
        db.save_message(chat_id, 'assistant', ai_response, intent=response_intent)
        
        # Log to Google Sheets
        log_to_sheets(
            chat_id, 
            user.get('first_name', 'Unknown'),
            text_message, 
            ai_response, 
            intent
        )
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Message processing error for {chat_id}: {e}")
        return "I'm experiencing technical difficulties. Please try again in a moment."


def send_whatsapp_message(chat_id: str, message: str) -> bool:
    """Send message via Green API"""
    try:
        url = f"https://api.green-api.com/waInstance{CONFIG['GREEN_API_ID']}/sendMessage/{CONFIG['GREEN_API_TOKEN']}"
        response = requests.post(
            url, 
            json={"chatId": chat_id, "message": message},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Send message error: {e}")
        return False


# Flask Routes
@app.route('/')
def health():
    return jsonify({
        "status": "healthy",
        "service": "AI Coach - Schoolinka",
        "timestamp": datetime.now().isoformat(),
        "pinecone_connected": pinecone_index is not None,
        "sheets_connected": users_sheet is not None and conversations_sheet is not None
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages - TEXT ONLY"""
    try:
        data = request.get_json()
        
        # Immediately acknowledge webhook
        if not data or data.get('typeWebhook') != 'incomingMessageReceived':
            return jsonify({"status": "ignored"}), 200
        
        message_data = data.get('messageData', {})
        sender_data = data.get('senderData', {})
        chat_id = sender_data.get('chatId', '').strip()
        
        if not chat_id:
            return jsonify({"status": "no_chat_id"}), 200
        
        # Extract TEXT message only
        text_message = None
        
        if 'textMessageData' in message_data:
            text_message = message_data['textMessageData'].get('textMessage', '').strip()
        elif 'extendedTextMessageData' in message_data:
            text_message = message_data['extendedTextMessageData'].get('text', '').strip()
        
        logger.info(f"Received text message from {chat_id}: {text_message[:50] if text_message else 'empty'}")
        
        if not text_message:
            # Ignore non-text messages
            send_whatsapp_message(chat_id, "I can only respond to text messages. Please type your question.")
            return jsonify({"status": "non_text_ignored"}), 200
        
        # Process in background thread
        def process_and_respond():
            try:
                reply = process_message(chat_id, text_message)
                send_whatsapp_message(chat_id, reply)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                send_whatsapp_message(chat_id, "Sorry, I encountered an error. Please try again.")
        
        thread = threading.Thread(target=process_and_respond)
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/user/<chat_id>', methods=['GET'])
def get_user_info(chat_id):
    """Get user profile and recent messages"""
    try:
        user = db.get_user(chat_id)
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404
        
        history = db.get_history(chat_id, limit=10)
        return jsonify({
            "status": "success",
            "user": user,
            "recent_messages": history
        })
    except Exception as e:
        logger.error(f"Error fetching user info: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/test', methods=['POST'])
def test():
    """Test endpoint for development"""
    try:
        data = request.get_json()
        chat_id = data.get('chat_id', 'test_user')
        message = data.get('message', 'Hello')
        
        response = process_message(chat_id, message)
        user = db.get_user(chat_id)
        
        return jsonify({
            "status": "success",
            "response": response,
            "user": user,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        with db.get_conn() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total_users FROM users')
            total_users = cursor.fetchone()['total_users']
            
            cursor.execute('SELECT COUNT(*) as total_messages FROM conversations')
            total_messages = cursor.fetchone()['total_messages']
            
            cursor.execute('SELECT COUNT(*) as registered_users FROM users WHERE profile_complete = 1')
            registered_users = cursor.fetchone()['registered_users']
        
        return jsonify({
            "status": "success",
            "stats": {
                "total_users": total_users,
                "registered_users": registered_users,
                "total_messages": total_messages,
                "pinecone_connected": pinecone_index is not None,
                "sheets_connected": users_sheet is not None
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting AI Coach - Schoolinka")
    logger.info(f"Pinecone Index: {'Connected' if pinecone_index else 'Not Available'}")
    logger.info(f"Google Sheets: {'Connected' if users_sheet and conversations_sheet else 'Not Available'}")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
