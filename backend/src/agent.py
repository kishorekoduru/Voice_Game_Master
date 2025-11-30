import logging
import json
import os
from typing import Annotated, Dict, Any, List
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.agents.llm import ChatMessage
from livekit.plugins import murf, silero, google, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("ecommerce-agent")

load_dotenv(".env.local")

# --- E-commerce Data & Logic ---

PRODUCTS = [
    {
        "id": "mug-001",
        "name": "Classic White Coffee Mug",
        "description": "A simple, elegant white ceramic mug. Perfect for your morning brew.",
        "price": 12.99,
        "currency": "USD",
        "category": "kitchen"
    },
    {
        "id": "mug-002",
        "name": "Travel Tumbler",
        "description": "Insulated stainless steel tumbler to keep your drinks hot or cold.",
        "price": 24.99,
        "currency": "USD",
        "category": "kitchen"
    },
    {
        "id": "shirt-001",
        "name": "Developer T-Shirt",
        "description": "Black cotton t-shirt with 'I turn coffee into code' print.",
        "price": 29.99,
        "currency": "USD",
        "category": "apparel",
        "sizes": ["S", "M", "L", "XL"]
    },
    {
        "id": "hoodie-001",
        "name": "Cozy Grey Hoodie",
        "description": "Soft fleece hoodie, perfect for coding sessions.",
        "price": 49.99,
        "currency": "USD",
        "category": "apparel",
        "sizes": ["M", "L", "XL"]
    }
]

ORDERS = []

class EcommerceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice shopping assistant for an online store.
            
            **Your Goal:** Help users find products and place orders.
            
            **Capabilities:**
            1. **Search/Browse:** Use `list_products` to find items based on user requests.
            2. **Order:** Use `create_order` when the user confirms they want to buy something.
            3. **History:** Use `get_last_order` if the user asks about their recent purchase.
            
            **Personality:**
            - Professional, polite, and efficient.
            - Keep responses concise (ideal for voice).
            - Confirm details before placing an order.
            
            **Flow:**
            - User asks for products -> You call `list_products` -> You summarize results.
            - User selects item -> You confirm details (quantity, etc.) -> You call `create_order`.
            - User asks "What did I buy?" -> You call `get_last_order`.
            """,
        )

    @function_tool
    async def list_products(self, ctx: RunContext, query: str = None, category: str = None):
        """List available products, optionally filtered by a search query or category.
        
        Args:
            query: Search term (e.g., "mug", "shirt").
            category: Filter by category (e.g., "kitchen", "apparel").
        """
        results = []
        for p in PRODUCTS:
            if category and p.get("category") != category:
                continue
            
            if query:
                q = query.lower()
                if q not in p["name"].lower() and q not in p["description"].lower():
                    continue
            
            results.append(p)
            
        return json.dumps(results)

    @function_tool
    async def create_order(self, ctx: RunContext, product_id: str, quantity: int = 1):
        """Place a new order for a product.
        
        Args:
            product_id: The ID of the product to buy.
            quantity: The number of items to purchase.
        """
        product = next((p for p in PRODUCTS if p["id"] == product_id), None)
        if not product:
            return "Error: Product not found."
            
        total_price = product["price"] * quantity
        
        order = {
            "order_id": f"ORD-{len(ORDERS) + 1:03d}",
            "product_id": product_id,
            "product_name": product["name"],
            "quantity": quantity,
            "total_price": total_price,
            "currency": product["currency"],
            "timestamp": datetime.now().isoformat()
        }
        
        ORDERS.append(order)
        
        # Optionally save to file
        try:
            with open("orders.json", "w") as f:
                json.dump(ORDERS, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save orders to file: {e}")
            
        return json.dumps(order)

    @function_tool
    async def get_last_order(self, ctx: RunContext):
        """Retrieve the most recent order placed in this session."""
        if not ORDERS:
            return "No orders found."
        return json.dumps(ORDERS[-1])


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    try:
        print(f"--- AGENT STARTING ---")
        ctx.log_context_fields = {"room": ctx.room.name}

        print(f"Connecting to room: {ctx.room.name}")
        await ctx.connect()
        print(f"--- AGENT CONNECTED to {ctx.room.name} ---")

        session = AgentSession(
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
            turn_detection=MultilingualModel(),
            vad=ctx.proc.userdata["vad"],
            preemptive_generation=True,
        )

        @session.on("agent_speech_committed")
        def on_agent_speech(msg: ChatMessage):
            print(f"--- AGENT SPEAKING: {msg.content} ---")
            
        @session.on("agent_speech_interrupted")
        def on_agent_interrupted(msg: ChatMessage):
            print(f"--- AGENT INTERRUPTED: {msg.content} ---")

        print("--- SESSION STARTING ---")
        await session.start(
            agent=EcommerceAgent(),
            room=ctx.room,
        )
        print("--- SESSION STARTED ---")

        await session.say("Hello! I'm your shopping assistant. How can I help you today?", allow_interruptions=True)

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
