from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label

from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
import faiss


class GhostMindUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.index = faiss.IndexFlatL2(384)
        self.memory_ids = []

        self.conn = sqlite3.connect('memory.db')
        self.cursor = self.conn.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                text TEXT
            )
        ''')

        self.conn.commit()

        self.search_input = TextInput(
            hint_text='Search your memories...'
        )

        self.search_button = Button(
            text='Search'
        )

        self.output = Label(text='GhostMind Ready')

        self.search_button.bind(on_press=self.search)

        self.add_widget(self.search_input)
        self.add_widget(self.search_button)
        self.add_widget(self.output)

        self.seed_demo_data()

    def embed(self, text):
        emb = self.model.encode([text])[0]
        return np.array([emb]).astype('float32')

    def add_memory(self, text):
        self.cursor.execute(
            'INSERT INTO memories (text) VALUES (?)',
            (text,)
        )

        self.conn.commit()

        memory_id = self.cursor.lastrowid

        embedding = self.embed(text)
        self.index.add(embedding)
        self.memory_ids.append(memory_id)

    def seed_demo_data(self):
        samples = [
            "Startup idea about offline AI memory",
            "Black hoodie from shopping screenshot",
            "Meeting notes about AI agents",
            "Recipe screenshot for pasta",
            "YouTube video about billion dollar startups",
            "Twitter thread on productivity systems",
            "Shopping screenshot for wireless headphones",
            "Screenshot of gym workout routine",
            "Idea for AI journaling app",
            "Flight ticket to Bangalore",
            "Crypto portfolio screenshot",
            "Book recommendation about psychology",
            "Startup funding strategy notes",
            "Dark mode UI inspiration screenshot",
            "Recipe for spicy ramen",
            "List of potential app names",
            "Screenshot of coding error in Python",
            "Voice memo about startup pitch",
            "Interesting article on neural networks",
            "Screenshot of premium sneaker collection",
            "Workout progress photo",
            "Reminder to renew domain name",
            "Screenshot of ChatGPT conversation",
            "AI agent architecture diagram",
            "Ideas for viral TikTok videos",
            "Screenshot of online course dashboard",
            "Meeting summary for product launch",
            "Screenshot of apartment listings",
            "Screenshot of startup landing page",
            "Business model ideas for SaaS",
            "Recipe for chocolate cake",
            "Screenshot of stock market chart",
            "Brainstorming notes about productivity app",
            "Travel itinerary for Goa trip",
            "Screenshot of cool app animation",
            "Shopping cart screenshot from Amazon",
            "Screenshot of Instagram growth tips",
            "AI-generated logo concepts",
            "Study notes about machine learning",
            "Screenshot of fitness app UI",
            "Ideas for offline-first mobile apps",
            "Screenshot of Linux terminal commands",
            "Meeting transcript with investor",
            "Screenshot of motivational quotes",
            "Design inspiration for dashboard",
            "Recipe screenshot for biryani",
            "Screenshot of coding tutorial",
            "Screenshot of browser tabs for research",
            "Notes about building AI agents",
            "Screenshot of startup metrics dashboard",
            "Screenshot of online banking transaction",
            "Screenshot of car modification ideas",
            "Voice note about marketing strategy",
            "Screenshot of futuristic UI design",
            "AI memory retrieval concept notes",
            "Screenshot of gaming setup inspiration",
            "Screenshot of smartwatch wishlist",
            "Screenshot of meditation techniques",
            "Shopping screenshot for mechanical keyboard",
            "Screenshot of healthy meal prep ideas",
            "Research notes on local AI models",
            "Screenshot of app onboarding screens",
            "Screenshot of design system colors",
            "Startup ideas involving wearable AI",
            "Screenshot of productivity hacks",
            "Screenshot of coding project structure",
            "Notes on edge AI computing",
            "Screenshot of startup valuation graph",
            "Recipe screenshot for protein smoothie",
            "Screenshot of task management workflow",
            "Screenshot of conference schedule",
            "Screenshot of cloud architecture diagram",
            "Voice memo about future business ideas",
            "Screenshot of cryptocurrency news",
            "Screenshot of mobile app wireframes",
            "Screenshot of AI chatbot discussion",
            "Screenshot of ecommerce product page",
            "Screenshot of JavaScript tutorial",
            "Notes on privacy-focused technology",
            "Screenshot of startup pitch deck",
            "Screenshot of book highlights",
            "Screenshot of machine learning roadmap",
            "Screenshot of inspirational workspace",
            "Screenshot of coding bootcamp curriculum",
            "Screenshot of UX research findings",
            "Screenshot of favorite quotes",
            "Screenshot of monthly budget spreadsheet",
            "Ideas for AI-powered search engine",
            "Screenshot of productivity planner",
            "Screenshot of offline navigation app",
            "Screenshot of hardware benchmarking results",
            "Screenshot of software architecture notes",
            "Recipe screenshot for homemade pizza",
            "Screenshot of AI-generated artwork",
            "Screenshot of startup growth chart",
            "Screenshot of deep learning article",
            "Screenshot of time management strategies",
            "Screenshot of remote work setup",
            "Screenshot of app monetization ideas",
            "Screenshot of futuristic gadget concepts",
            "Screenshot of open source AI tools",
            "Screenshot of business analytics dashboard",
            "Screenshot of brainstorming whiteboard",
            "Screenshot of startup competitor analysis",
            "Screenshot of AI voice assistant concepts"
        ]

        for s in samples:
            self.add_memory(s)

    def search(self, instance):
        query = self.search_input.text

        if not query:
            return

        q = self.embed(query)

        distances, indices = self.index.search(q, 3)

        results = []

        for idx in indices[0]:
            if idx >= len(self.memory_ids):
                continue

            memory_id = self.memory_ids[idx]

            self.cursor.execute(
                'SELECT text FROM memories WHERE id=?',
                (memory_id,)
            )

            row = self.cursor.fetchone()

            if row:
                results.append(row[0])

        self.output.text = '\n'.join(results)


class GhostMindApp(App):
    def build(self):
        return GhostMindUI()


GhostMindApp().run()
