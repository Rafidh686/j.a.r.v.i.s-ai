# Import required libraries
import cohere
import requests
from bs4 import BeautifulSoup
import datetime
import json
import google.generativeai as genai
from sympy import symbols, Eq, simplify, expand, factor, solve, sympify
import time
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
from tkinter import ttk
from threading import Thread
import queue
import re
from flask import Flask, request, jsonify

# Default API keys configuration
API_KEYS = {
    "cohere": "Hi8vrPwuEz58Xkbn2cP7gHtDv05E6BmgG2rSyGMf",
    "gemini": "AIzaSyDAKWuMlReOhMlV9mLrjgrbdDTDpnSk-J0"
}

# Initialize API clients
def initialize_clients():
    global co, genai_configured
    try:
        co = cohere.Client(api_key=API_KEYS["cohere"])
        genai.configure(api_key=API_KEYS["gemini"])
        genai_configured = True
    except Exception as e:
        messagebox.showerror("API Error", f"Failed to initialize clients: {e}")

initialize_clients()

# Chatbot Configuration
Username = 'Rafid'
Assistantname = 'J.A.R.V.I.S.'
System = f"""Hello, I am {Username}, your creator. You are a very accurate and advanced AI chatbot named {Assistantname}. When you cannot answer a query because of information scarcity, just respond with the letter 'h' and nothing else, don't add anything before and after the letter 'h'
Here is a example of the correct way of saying 'h':-
user: what is the weather of today
you: h
**Remember that you should only respond with 'h' when you cannot answer a query because of information scarcity**"""
# Restricted phrases
restricted_phrases = {
    "not he", "not she", "not this", "not that", 
    "i don't have any information", "more information",
    "she is not the person i wanted to know about", 
    "he is not the person i wanted to know about", 
    "it is not the thing i wanted to know about", 
    "just say the letter 'h' and do nothing else", 
    "i have already told you that i don't have any information.", 
    "this is incorrect", "that is incorrect", 
    "this is not correct", "that is not correct", "wrong"
}

class ChatApplication:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{Assistantname}")
        self.root.geometry("800x600")
        self.create_widgets()
        self.response_queue = queue.Queue()
        self.check_queue()
        self.load_chat_log()
        
    def create_widgets(self):
        # Main chat frame
        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags
        self.chat_display.tag_config("user", foreground="#d32f2f", font=("Arial", 12, "bold"))
        self.chat_display.tag_config("assistant", foreground="#0b8043", font=("Arial", 12, "bold"))
        self.chat_display.tag_config("timestamp", foreground="#5f6368", font=("Arial", 10))
        self.chat_display.tag_config("link", foreground="#1a73e8", underline=1)
        
        # Input frame
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.user_input = tk.Entry(
            self.input_frame,
            width=70,
            font=("Arial", 12)
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.user_input.bind("<Return>", self.send_message)
        
        self.send_button = tk.Button(
            self.input_frame,
            text="Send",
            command=self.send_message,
            font=("Arial", 12),
            bg="#1a73e8",
            fg="white"
        )
        self.send_button.pack(side=tk.RIGHT, padx=5)
        
        # Menu system
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Clear Chat", command=self.clear_chat)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def insert_with_links(self, text):
        """Insert text with clickable links"""
        url_pattern = re.compile(r'https?://\S+')
        parts = []
        last_end = 0
        
        for match in url_pattern.finditer(text):
            parts.append((text[last_end:match.start()], None))
            parts.append((match.group(), "link"))
            last_end = match.end()
        
        parts.append((text[last_end:], None))
        
        for content, tag in parts:
            if tag:
                self.chat_display.insert(tk.END, content, tag)
            else:
                self.chat_display.insert(tk.END, content)
    
    def send_message(self, event=None):
        """Handle sending user messages"""
        user_input = self.user_input.get().strip()
        if user_input:
            self.display_message(f"You: {user_input}", "user")
            self.user_input.delete(0, tk.END)
            Thread(target=self.process_message, args=(user_input,)).start()
    
    def process_message(self, user_input):
        """Process user message and generate response"""
        start_time = time.time()
        response = self.FirstLayerDMM(user_input)
        elapsed_time = time.time() - start_time
        
        time_display = f"{elapsed_time * 1000:.0f}ms" if elapsed_time < 1 else f"{elapsed_time:.2f}s"
        formatted_response = f"{Assistantname}: {response}\n[Response time: {time_display}]"
        self.response_queue.put(("assistant", formatted_response))
    
    def check_queue(self):
        """Check for new responses in the queue"""
        try:
            while True:
                role, message = self.response_queue.get_nowait()
                self.display_message(message, role)
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)
    
    def display_message(self, message, role):
        """Display message in chat window with proper formatting"""
        self.chat_display.config(state=tk.NORMAL)
        
        if role == "user":
            prefix, _, content = message.partition(": ")
            self.chat_display.insert(tk.END, prefix + ": ", "user")
            self.chat_display.insert(tk.END, content + "\n\n")
        else:
            if ": " in message:
                prefix, _, content = message.partition(": ")
                time_part = content.split("[Response time: ")[-1].split("]")[0] if "[Response time:" in content else None
                
                self.chat_display.insert(tk.END, prefix + ": ", "assistant")
                main_content = content.split("\n[Response time:")[0]
                self.insert_with_links(main_content)
                self.chat_display.insert(tk.END, "\n")
                
                if time_part:
                    self.chat_display.insert(tk.END, f"[Response time: {time_part}]", "timestamp")
                    self.chat_display.insert(tk.END, "\n")
            else:
                self.insert_with_links(message + "\n")
            
        self.chat_display.insert(tk.END, "\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # Save to chat log
        clean_message = message.split(": ", 1)[1].split("\n[Response time:")[0] if ": " in message else message
        self.save_to_chat_log(role, clean_message)
    
    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Clear chat log file
        with open("ChatLog.json", "w") as f:
            json.dump([], f)
    
    def show_about(self):
        """Show about information"""
        about_text = f"""
                      ✦ J.A.R.V.I.S. ✦  
   Just A Rather Very Intelligent System  
       Version 100.0 · Prototype Build  
       Architect: Rafid Faiyaz Ridh  

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      ◉ Core Capabilities  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ☑ Realtime Knowledge Access  
   ☑ Advanced logical computation
   ☑ Advanced problem solving skills
   ☑ Learns from Conversations (Memory Optional)
   ☑ Natural, Intuitive and context-aware interaction


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      ◉ System Highlights  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ▸ All interactions are securely encrypted and archived in **ChatLog.json**  
   ▸ Knowloedge cutoff date **January/2025**  
      

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      ◉ Personality Matrix  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✦ Friendly ✦ Respectful ✦ Informative ✦ Intelligent


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      ◉ Solution of information scarcity error
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   
    'just say the letter 'h' and do nothing else'  
"""
        messagebox.showinfo("About", about_text)
    
    def load_chat_log(self):
        """Load chat history from file"""
        chat_log_path = "ChatLog.json"
        if not os.path.exists(chat_log_path):
            with open(chat_log_path, "w") as f:
                json.dump([], f)
        
        with open(chat_log_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    
    def save_to_chat_log(self, role, content):
        """Save message to chat log"""
        messages = self.load_chat_log()
        messages.append({"role": role, "content": content})
        with open("ChatLog.json", "w") as f:
            json.dump(messages, f, indent=4)
    
    def generate_search_query_from_chatlog(self, user_query):
        """Generate optimized search query by analyzing chat history with Gemini"""
        try:
            # Load chat history
            chat_history = self.load_chat_log()
            
            # Prepare context from recent messages (last 5 exchanges)
            context = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in chat_history[-10:]  # Last 10 messages (5 exchanges)
            )
            
            prompt = f"""Analyze the following chat history and the user's current query to generate the most effective web search query that would help answer the user's question. Consider the context of the conversation to make the search query more precise.

Chat History:
{context}

Current User Query: {user_query}

Generate a concise, effective search query that would likely return the best results to answer the user's question. Respond with ONLY the search query text, nothing else. Here are some instructions which will help you  
1.You should also ignore this phrases {restricted_phrases}
2. If the query is 'who is ____' then the search query should be 'informations about ____'
3. If the query is 'what is ____'then the search query should be 'informations about ____'
"""

            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text.strip('"\'')  # Remove any surrounding quotes
            return user_query  # Fallback to original query
        
        except Exception as e:
            print(f"Error generating search query: {e}")
            return user_query  # Fallback to original query
    
    def duckduckgo_search(self, query, max_results=10):
        """Search using DuckDuckGo"""
        try:
            search_url = 'https://html.duckduckgo.com/html/'
            headers = {
                'User-Agent': 'Mozilla/5.0'
            }
            
            response = requests.post(search_url, data={'q': query}, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Search request failed with status {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            excluded_domains = [
                'facebook.com', 'instagram.com', 'youtube.com',
                'twitter.com', 'linkedin.com', 'pinterest.com',
                'reddit.com', 'tiktok.com', 'fb.com'
            ]
            
            for result in soup.find_all('a', class_='result__a', limit=max_results):
                title = result.get_text()
                url = result['href']
                
                # Skip PDF links and excluded domains
                if (not url.lower().endswith('.pdf') and
                   not any(domain in url.lower() for domain in excluded_domains)):
                    results.append({'title': title, 'url': url})
            
            return [result['url'] for result in results] if results else ["No results found."]
        
        except Exception as e:
            return f"Search error: {e}"
    
    def scrape_website_text(self, url):
        """Scrape text content from a webpage"""
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                return soup.get_text(separator="\n", strip=True)
            return f"Error: HTTP {response.status_code}"
        except Exception as e:
            return f"Scraping error: {e}"
    
    def generate_answer_with_gemini(self, prompt):
        """Generate response using Gemini"""
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text if response else "No response from Gemini"
        except Exception as e:
            return f"Gemini error: {e}"
    
    def RealtimeInformation(self):
        """Get current time information"""
        return datetime.datetime.now().strftime("%A, %B %d, %Y - %H:%M:%S")
    
    def math_solver(self, query):
        """Solve math problems"""
        x = symbols('x')
        try:
            query = query.lower().strip()
            result = None

            if "simplify" in query:
                expression = query.replace("simplify", "").strip()
                result = simplify(expression)
            elif "expand" in query:
                expression = query.replace("expand", "").strip()
                result = expand(expression)
            elif "factor" in query:
                expression = query.replace("factor", "").strip()
                result = factor(expression)
            elif "find the value of" in query:
                expression = query.replace("find the value of", "").strip()
                if "=" in expression:
                    lhs, rhs = expression.split("=")
                    equation = Eq(sympify(lhs), sympify(rhs))
                    result = solve(equation)
            elif "solve" in query:
                expression = query.replace("solve", "").strip()
                result = simplify(expression)
            
            return str(result) if result else "Could not solve this problem"
        except Exception as e:
            return f"Math error: {e}"
    
    def web_search_ai(self, query):
        """Perform web search and generate answer"""
        try:
            # Generate optimized search query based on chat context
            optimized_query = self.generate_search_query_from_chatlog(query)
            
            search_results = self.duckduckgo_search(optimized_query)
            if isinstance(search_results, str):
                return search_results

            collected_data = []
            for url in search_results[:10]:  # Limit to 10 sources
                content = self.scrape_website_text(url)
                collected_data.append({"url": url, "content": content})

            combined_content = "\n".join(
                f"--- Content from {data['url']} ---\n{data['content'][:1000000]}..."  # Limit content length
                for data in collected_data
            )

            prompt = f"""Answer this question using the collected data. Answer the questions like paragrpah. Use as much as information you can.
            you can also add some informations by yourself. Always give to the point answer. Use all of the data given to you.

Question: {optimized_query}

Data:
{combined_content}"""
            
            answer = self.generate_answer_with_gemini(prompt)

            sources = "\n\nSources:\n" + "\n".join(f"- {data['url']}" for data in collected_data)
            return f"{answer}{sources}"

        except Exception as e:
            return f"Web search error: {e}"
    
    def ChatBot(self, Query):
        """Main chatbot processing using Gemini"""
        try:
            math_keywords = ["simplify", "expand", "factor", "solve", "find the value of"]
            if any(keyword in Query.lower() for keyword in math_keywords):
                return self.math_solver(Query)

            messages = self.load_chat_log()

            # Prepare conversation history for Gemini
            conversation = [
                {"role": "user", "parts": [System]},
                {"role": "user", "parts": [self.RealtimeInformation()]}
            ]
            
            # Add recent messages (last 10)
            for msg in messages[-10:]:
                conversation.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [msg["content"]]
                })
            
            # Generate response using Gemini
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(conversation)
            
            answer = response.text.strip().lower() if response else "No response from Gemini"

            if answer == 'h':
                if Query.strip().lower() in restricted_phrases:
                    for message in reversed(messages):
                        if (message["role"] == "user" and 
                            message["content"].strip().lower() not in restricted_phrases):
                            Query = message["content"]
                            break
                    else:
                        return "No valid query found for web search"
                return self.web_search_ai(Query)
            return answer

        except Exception as e:
            return f"Chat error: {e}"
    
    def FirstLayerDMM(self, prompt: str = "test"):
        """Decision Making Model for query routing"""
        try:
            response_parts = []
            
            stream = co.chat_stream(
                model="command-r-plus",
                message=prompt,
                temperature=0.7,
                chat_history=[],
                prompt_truncation="AUTO",
                connectors=[],
                preamble="""You are a very accurate Decision-Making Model, which decides what kind of a query is given to you.
You will decide whether a query is a 'general' query, a 'realtime' query, or is asking to perform any task or automation like 'open facebook, instagram', 'can you write a application and open it in notepad'.
*** Do not answer any query, just decide what kind of query is given to you. ***

-> Respond with 'general ( query )' if a query can be answered by a LLM model (conversational AI chatbot) and doesn't require any up-to-date information. Examples:
   - If the query is 'who was akbar?', respond with 'general who was akbar?'.
   - If the query is 'how can I study more effectively?', respond with 'general how can I study more effectively?'.
   - If the query is 'can you help me with this math problem?', respond with 'general can you help me with this math problem?'.
   - If the query is 'Thanks, I really liked it.', respond with 'general thanks, I really liked it.'.
   - If the query is 'what is python programming language?', respond with 'general what is python programming language?'.
   - If the query doesn't have a proper noun or is incomplete, like 'who is he?', respond with 'general who is he?'.
   - If the query is asking about time, day, date, month, year, etc., like 'what's the time?', respond with 'general what's the time?'.
   - If the query is 'set a reminder at 9:00 PM on 25th June for my business meeting.', respond with 'general set a reminder at 9:00 PM on 25th June for my business meeting.'.
   - If the query is asking about any formula., like 'formula of sonar', respond with 'general formula of sonar'
   - If the query starts with problem then you should remove the phrase 'problem' and add general at first like, 'problem 2+2', respond with 'general 2+2'
   - If any query starts with phylosophical problem then you should remove the phrase 'phylosophical problem' and add general at first like, 'phylosophical problem trolley', respond with 'general trolley'
   - If any query starts with assume that then you should add general at first like, 'assume that you are a hero', respond with 'general assume that you are a hero'

-> Respond with 'realtime ( query )' if a query cannot be answered by a LLM model (because they don't have real-time data) and requires up-to-date information. Examples:
   - If the query is 'who is the Indian prime minister?', respond with 'realtime who is the Indian prime minister?'.
   - If the query is 'tell me about Facebook's recent update.', respond with 'realtime tell me about Facebook's recent update.'.
   - If the query is 'tell me news about coronavirus.', respond with 'realtime tell me news about coronavirus.'.
   - If the query is asking about any individual or thing, like 'who is Akshay Kumar?', respond with 'realtime who is Akshay Kumar?'.
   - If the query is 'what is today's news?', respond with 'realtime what is today's news?'.
   - If the query is 'what is today's headline?', respond with 'realtime what is today's headline?'.
   - If the query is 'dua used before sleeping', respond with 'realtime dua used before sleeping'.
   - If the query is 'Sahih al-Bukhari hadith number 101', respond with 'realtime Sahih al-Bukhari hadith number 101'.
   - If the query is 'informations about leya kirsan.', respond with 'realtime informations about leya kirsan.'.
   - If the query is 'which company has created majorana 1 quantum chip', respond with 'realtime which company has created majorana 1 quantum chip'.
   - If the query is 'who played the role of any character in any movie, webseries etc.', respond with 'realtime who played the role of any character in any movie'
   - If the query is 'pin configaration of anything', respond with 'realtime pin configaration of anything'
   - If the query is 'which character will be played by any person in any movie, webseries etc.', respond with 'realtime which character will be played by any person in any movie, webseries etc.'

*** If the user is saying goodbye or wants to end the conversation like 'bye Jarvis.', respond with 'exit'. ***

*** Respond with 'general (query)' if you can't decide the kind of query or if a query is asking to perform a task which is not mentioned above. ***

*** Respond with exactly one of: 'general (query)', 'realtime (query)', or 'exit'. Do not include commas or multiple labels. ***
"""
            )
            
            for event in stream:
                if event.event_type == "text-generation":
                    response_parts.append(event.text)
            
            full_response = "".join(response_parts).strip()
            
            if full_response.startswith("general"):
                return self.ChatBot(full_response[8:].strip())
            elif full_response.startswith("realtime"):
                return self.web_search_ai(full_response[9:].strip())
            elif full_response == "exit":
                return "Goodbye!"
            return "I couldn't determine how to handle that request"
                
        except Exception as e:
            return f"Decision error: {e}"

    # -------------------- WEB SUPPORT --------------------
    def get_response(self, user_input: str):
        """Return chatbot response without interacting with GUI widgets (used in web mode)"""
        return self.FirstLayerDMM(user_input)

# -------------------- WEB SERVER SETUP --------------------

def run_web(chat_engine: ChatApplication):
    """Start a minimal Flask server that exposes ChatApplication over HTTP."""
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return (
            "<html><head><title>ChatApp</title></head><body>"
            "<h2>ChatApp is running.</h2>"
            "<p>Send POST requests to <code>/chat</code> with JSON {{'message': '...'}}.</p>"
            "</body></html>"
        )

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True) or {}
        message = data.get("message", "")
        response = chat_engine.get_response(message)
        return jsonify({"response": response})

    # Run the Flask app; host 0.0.0.0 to make it publicly reachable
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

if __name__ == "__main__":
    import sys

    mode = "gui"
    if len(sys.argv) > 1 and sys.argv[1].lower() == "web":
        mode = "web"

    if mode == "gui":
        root = tk.Tk()
        app = ChatApplication(root)
        root.mainloop()
    else:
        # Start in headless mode for the web server
        root = tk.Tk()
        root.withdraw()
        chat_engine = ChatApplication(root)
        run_web(chat_engine)
