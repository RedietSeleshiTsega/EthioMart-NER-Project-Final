from telethon.sync import TelegramClient
from telethon.tl.types import PeerChannel
import pandas as pd
import os

api_id = '27255481'
api_hash = '991e24e0f5f018fa08daa6d11dbf5962'
phone = '+251934967474'

client = TelegramClient('ethio_scraper', api_id, api_hash)

async def scrape_channel(channel_usernames, limit=200):
    await client.start()
    all_data = []

    for username in channel_usernames:
        try:
            print(f"Fetching from: {username}")
            async for message in client.iter_messages(username, limit=limit):
                if message.text:
                    all_data.append({
                        'channel': username,
                        'message': message.text,
                        'date': str(message.date),
                        'sender_id': str(message.sender_id)
                    })
        except Exception as e:
            print(f"Error fetching {username}: {e}")

    df = pd.DataFrame(all_data)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/telegram_data.csv", index=False)
    print("✅ Data saved to data/raw/telegram_data.csv")

if __name__ == '__main__':
    import asyncio
    channels = [
        '@ZemenExpress', '@nevacomputer', '@meneshayeofficial',
        '@ethio_brand_collection', '@Leyueqa', '@sinayelj',
        '@Shewabrand', '@helloomarketethiopia', '@modernshoppingcenter',
        '@qnashcom', '@Fashiontera', '@kuruwear', '@gebeyaadama',
        '@MerttEka', '@forfreemarket', '@classybrands', '@marakibrand',
        '@aradabrand2', '@marakisat2', '@belaclassic', '@AwasMart'
    ]
    asyncio.run(scrape_channel(channels, limit=200))