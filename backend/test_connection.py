import os
import asyncio
from livekit import api
from dotenv import load_dotenv

load_dotenv(".env.local")

async def main():
    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")

    print(f"Connecting to {url}...")
    try:
        lkapi = api.LiveKitAPI(url, key, secret)
        rooms = await lkapi.room.list_rooms(api.ListRoomsRequest())
        print(f"Successfully connected! Found {len(rooms.rooms)} rooms.")
        for room in rooms.rooms:
            print(f" - Room: {room.name}")
        await lkapi.aclose()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
