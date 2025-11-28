import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import ChatMessage
from livekit.plugins import google, deepgram, silero

load_dotenv(".env.local")
logger = logging.getLogger("minimal-agent")

class MinimalAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a minimal agent.")

import traceback

async def entrypoint(ctx: JobContext):
    try:
        print(f"--- AGENT STARTING ---")
        print(f"Connecting to room: {ctx.room.name}")
        
        await ctx.connect()
        print(f"--- AGENT CONNECTED to {ctx.room.name} ---")

        # Check if there are other participants
        # print(f"Participants: {ctx.room.participants}")

        print("--- SESSION STARTING ---")
        session = AgentSession(
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=google.TTS(),
            vad=silero.VAD.load(),
        )

        @session.on("user_speech_committed")
        def on_speech_committed(msg: ChatMessage):
            print(f"--- SPEECH DETECTED: {msg.content} ---")

        @session.on("agent_speech_committed")
        def on_agent_speech(msg: ChatMessage):
            print(f"--- AGENT SPEAKING: {msg.content} ---")

        await session.start(
            agent=MinimalAgent(),
            room=ctx.room,
        )
        print("--- SESSION STARTED ---")
        
        print("--- SAYING HELLO ---")
        await session.say("Hello! I am a minimal agent. Can you hear me?", allow_interruptions=True)
        print("--- SAID HELLO ---")
    except Exception:
        print("--- CRASH DETECTED ---")
        traceback.print_exc()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
