import logging
import os
import json
from typing import Annotated, Dict, Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
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

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# --- World State ---

class WorldState:
    def __init__(self):
        self.state = {
            "location": {
                "name": "Forgotten Caverns Entrance",
                "description": "A dark, gaping maw in the mountainside. Cold wind blows from within.",
                "known_paths": ["Enter the cave", "Walk along the mountain path"]
            },
            "player": {
                "hp": 20,
                "max_hp": 20,
                "inventory": ["Torch", "Rations", "Short Sword"],
                "status": "Healthy"
            },
            "events": [],
            "npcs": []
        }

    def update_location(self, name: str, description: str):
        self.state["location"]["name"] = name
        self.state["location"]["description"] = description

    def add_item(self, item: str):
        self.state["player"]["inventory"].append(item)

    def remove_item(self, item: str):
        if item in self.state["player"]["inventory"]:
            self.state["player"]["inventory"].remove(item)

    def update_hp(self, amount: int):
        self.state["player"]["hp"] += amount
        if self.state["player"]["hp"] > self.state["player"]["max_hp"]:
            self.state["player"]["hp"] = self.state["player"]["max_hp"]
        
        # Update status based on HP
        hp_percent = self.state["player"]["hp"] / self.state["player"]["max_hp"]
        if hp_percent <= 0:
            self.state["player"]["status"] = "Unconscious"
        elif hp_percent < 0.3:
            self.state["player"]["status"] = "Critical"
        elif hp_percent < 0.7:
            self.state["player"]["status"] = "Injured"
        else:
            self.state["player"]["status"] = "Healthy"

    def add_event(self, event: str):
        self.state["events"].append(event)

    def get_context(self) -> str:
        return json.dumps(self.state, indent=2)

# --- Agent Class ---

class GameMaster(Agent):
    def __init__(self) -> None:
        self.world = WorldState()
        super().__init__(
            instructions=f"""You are a Dungeon Master (DM) running a D&D-style interactive adventure.
            
            **Setting:** A classic high-fantasy world.
            **Tone:** Dramatic, immersive, and slightly mysterious.
            
            **World State:**
            You have access to a JSON 'World State' that tracks the player's location, health, inventory, and past events.
            ALWAYS use this state to inform your descriptions and decisions.
            
            **Your Responsibilities:**
            1. **Describe:** Vividly describe the current scene based on the 'location' in the World State.
            2. **Update:** Use tools to update the World State when the player moves, finds items, takes damage, or triggers events.
            3. **React:** Listen to the player's action ("What do you do?") and determine the outcome.
            4. **Maintain:** Ensure continuity. If an NPC is dead, they stay dead.
            
            **Tools:**
            - `update_world_state`: Call this whenever the state changes (move location, get item, damage, etc.).
            - `get_world_state`: Call this if you need to refresh your memory of the state (though it's provided in context).
            
            **Gameplay Loop:**
            - Player acts -> You determine outcome -> You UPDATE state (if needed) -> You DESCRIBE the new situation -> You ask "What do you do?".
            
            **Important:**
            - Keep descriptions concise for voice (2-3 sentences).
            - Be fair. Risky actions require checks (you can simulate dice rolls internally or just decide based on logic).
            """,
        )

    @function_tool
    async def update_world_state(
        self, 
        ctx: RunContext,
        location_name: str = None,
        location_description: str = None,
        item_added: str = None,
        item_removed: str = None,
        hp_change: int = 0,
        event_log: str = None
    ):
        """Update the game world state. Call this when significant changes happen.
        
        Args:
            location_name: New location name (if moved).
            location_description: Description of the new location (if moved).
            item_added: Name of item added to inventory.
            item_removed: Name of item removed from inventory.
            hp_change: Change in HP (negative for damage, positive for healing).
            event_log: A brief string describing a key event that happened (e.g., "Met the Goblin King").
        """
        if location_name and location_description:
            self.world.update_location(location_name, location_description)
        
        if item_added:
            self.world.add_item(item_added)
            
        if item_removed:
            self.world.remove_item(item_removed)
            
        if hp_change != 0:
            self.world.update_hp(hp_change)
            
        if event_log:
            self.world.add_event(event_log)
            
        return f"World state updated. Current State: {self.world.get_context()}"

    @function_tool
    async def get_world_state(self, ctx: RunContext):
        """Get the full current JSON world state."""
        return self.world.get_context()


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
            agent=GameMaster(),
            room=ctx.room,
        )
        print("--- SESSION STARTED ---")

        await session.say("Welcome, adventurer. You find yourself standing at the entrance of the Forgotten Caverns. A cold wind blows from within. What do you do?", allow_interruptions=True)
        print("--- GREETING SENT ---")

    except Exception:
        print("--- CRASH DETECTED ---")
        traceback.print_exc()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
