import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    llm,
    function_tool,
    RunContext
)
from livekit.agents.llm import ChatMessage
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# --- Data Models & Helper Classes ---

class CartItem:
    def __init__(self, item_id: str, name: str, price: float, quantity: int = 1, notes: str = ""):
        self.item_id = item_id
        self.name = name
        self.price = price
        self.quantity = quantity
        self.notes = notes

    def to_dict(self):
        return {
            "item_id": self.item_id,
            "name": self.name,
            "price": self.price,
            "quantity": self.quantity,
            "notes": self.notes,
            "subtotal": self.price * self.quantity
        }

class Cart:
    def __init__(self):
        self.items: Dict[str, CartItem] = {}

    def add_item(self, item_id: str, name: str, price: float, quantity: int = 1, notes: str = ""):
        if item_id in self.items:
            self.items[item_id].quantity += quantity
            if notes:
                self.items[item_id].notes = f"{self.items[item_id].notes}; {notes}".strip("; ")
        else:
            self.items[item_id] = CartItem(item_id, name, price, quantity, notes)

    def remove_item(self, item_id: str):
        if item_id in self.items:
            del self.items[item_id]

    def update_quantity(self, item_id: str, quantity: int):
        if item_id in self.items:
            if quantity <= 0:
                del self.items[item_id]
            else:
                self.items[item_id].quantity = quantity

    def clear(self):
        self.items = {}

    def get_total(self) -> float:
        return sum(item.price * item.quantity for item in self.items.values())

    def to_dict(self):
        return {
            "items": [item.to_dict() for item in self.items.values()],
            "total": self.get_total()
        }

    def is_empty(self):
        return len(self.items) == 0

# --- Catalog Helper ---

def load_catalog():
    try:
        with open("catalog.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("catalog.json not found!")
        return {"categories": []}

CATALOG = load_catalog()

def find_item_by_name(name: str):
    name_lower = name.lower()
    for category in CATALOG.get("categories", []):
        for item in category.get("items", []):
            if name_lower in item["name"].lower():
                return item
    return None

def find_items_by_tag(tag: str):
    items = []
    for category in CATALOG.get("categories", []):
        for item in category.get("items", []):
            if tag.lower() in [t.lower() for t in item.get("tags", [])]:
                items.append(item)
    return items

# --- Agent Class ---

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and helpful food & grocery ordering assistant for 'QuickMart'.
            Your goal is to help users browse the catalog, add items to their cart, and place orders.
            
            Capabilities:
            - You can list available items from the catalog.
            - You can add items to the cart. If a user asks for "ingredients for X", try to add the relevant items (e.g., bread + peanut butter for a sandwich).
            - You can remove items or update quantities.
            - You can show the current cart status.
            - You can place the order when the user is ready.
            
            Behavior:
            - Be polite and concise.
            - Confirm actions verbally (e.g., "I've added 2 apples to your cart.").
            - If an item is not found, apologize and suggest alternatives if possible.
            - When placing an order, summarize the total and ask for final confirmation if you haven't already.
            
            Catalog Context:
            The catalog is loaded in the system. You can query it using your tools or general knowledge if provided in context.
            """,
        )
        self.cart = Cart()

    @function_tool
    async def get_catalog(self, ctx: RunContext):
        """Get the list of available categories and items in the catalog.
        
        Returns:
            str: The full catalog structure in JSON format.
        """
        return json.dumps(CATALOG)

    @function_tool
    async def add_to_cart(
        self, 
        ctx: RunContext,
        item_name: str,
        quantity: int = 1,
        notes: str = ""
    ):
        """Add an item to the shopping cart.
        
        Args:
            item_name: The name of the item to add.
            quantity: The quantity to add.
            notes: Any special notes or preferences.
        """
        item = find_item_by_name(item_name)
        if item:
            self.cart.add_item(item["id"], item["name"], item["price"], quantity, notes)
            return f"Added {quantity} x {item['name']} to cart."
        else:
            return f"Sorry, I couldn't find '{item_name}' in the catalog."

    @function_tool
    async def add_ingredients_for_meal(
        self,
        ctx: RunContext,
        meal_name: str
    ):
        """Add multiple items needed for a recipe or meal.
        
        Args:
            meal_name: The name of the meal (e.g., 'peanut butter sandwich', 'pasta').
        """
        added_items = []
        if "sandwich" in meal_name.lower() and "peanut butter" in meal_name.lower():
            # Example logic for PB&J
            pb = find_item_by_name("Peanut Butter")
            bread = find_item_by_name("Bread")
            if pb: self.cart.add_item(pb["id"], pb["name"], pb["price"], 1)
            if bread: self.cart.add_item(bread["id"], bread["name"], bread["price"], 1)
            added_items = ["Peanut Butter", "Bread"]
        elif "pasta" in meal_name.lower():
            pasta = find_item_by_name("Pasta")
            sauce = find_item_by_name("Sauce")
            if pasta: self.cart.add_item(pasta["id"], pasta["name"], pasta["price"], 1)
            if sauce: self.cart.add_item(sauce["id"], sauce["name"], sauce["price"], 1)
            added_items = ["Pasta", "Marinara Sauce"]
        
        if added_items:
            return f"Added ingredients for {meal_name}: {', '.join(added_items)}."
        else:
            return f"I'm not sure what ingredients are needed for {meal_name}. Please add items individually."

    @function_tool
    async def remove_from_cart(
        self,
        ctx: RunContext,
        item_name: str
    ):
        """Remove an item from the cart.
        
        Args:
            item_name: The name of the item to remove.
        """
        item = find_item_by_name(item_name)
        if item and item["id"] in self.cart.items:
            self.cart.remove_item(item["id"])
            return f"Removed {item['name']} from cart."
        return f"Item '{item_name}' not found in cart."

    @function_tool
    async def get_cart_status(self, ctx: RunContext):
        """Get the current status of the shopping cart.
        
        Returns:
            str: The current items in the cart and the total price.
        """
        if self.cart.is_empty():
            return "Your cart is empty."
        return json.dumps(self.cart.to_dict())

    @function_tool
    async def place_order(self, ctx: RunContext):
        """Place the final order.
        
        Returns:
            str: Confirmation message with order ID and total.
        """
        if self.cart.is_empty():
            return "Your cart is empty. I cannot place an order."
        
        order_data = {
            "order_id": f"ORD-{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "status": "received",
            "items": [item.to_dict() for item in self.cart.items.values()],
            "total": self.cart.get_total()
        }
        
        # Save to orders.json
        try:
            orders = []
            if os.path.exists("orders.json"):
                with open("orders.json", "r") as f:
                    try:
                        orders = json.load(f)
                    except json.JSONDecodeError:
                        orders = []
            
            orders.append(order_data)
            
            with open("orders.json", "w") as f:
                json.dump(orders, f, indent=2)
                
            self.cart.clear()
            return f"Order placed successfully! Order ID: {order_data['order_id']}. Total: ${order_data['total']:.2f}"
        except Exception as e:
            logger.error(f"Failed to save order: {e}")
            return "There was an error saving your order. Please try again."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


import traceback

async def entrypoint(ctx: JobContext):
    try:
        print(f"--- AGENT STARTING ---")
        # Logging setup
        ctx.log_context_fields = {
            "room": ctx.room.name,
        }

        print(f"Connecting to room: {ctx.room.name}")
        # Join the room and connect to the user
        await ctx.connect()
        print(f"--- AGENT CONNECTED to {ctx.room.name} ---")

        # Set up a voice AI pipeline
        session = AgentSession(
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(
                    model="gemini-2.5-flash",
                ),
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
            print(f"--- AGENT SPEAKING (Murf): {msg.content} ---")
            
        @session.on("agent_speech_interrupted")
        def on_agent_interrupted(msg: ChatMessage):
            print(f"--- AGENT INTERRUPTED: {msg.content} ---")

        usage_collector = metrics.UsageCollector()

        @session.on("metrics_collected")
        def _on_metrics_collected(ev: MetricsCollectedEvent):
            metrics.log_metrics(ev.metrics)
            usage_collector.collect(ev.metrics)

        async def log_usage():
            summary = usage_collector.get_summary()
            logger.info(f"Usage: {summary}")

        ctx.add_shutdown_callback(log_usage)

        print("--- SESSION STARTING ---")
        # Start the session
        await session.start(
            agent=Assistant(),
            room=ctx.room,
        )
        print("--- SESSION STARTED ---")

        await session.say("Hello! I am connected and ready to take your order.", allow_interruptions=True)
        print("--- GREETING SENT ---")

    except Exception:
        print("--- CRASH DETECTED ---")
        traceback.print_exc()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
