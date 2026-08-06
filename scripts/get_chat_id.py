"""
Run this once to get your Telegram chat ID.

    python scripts/get_chat_id.py

Then send /start to your bot in Telegram.
It will reply with your chat ID — copy it into .env as TELEGRAM_CHAT_ID=<id>
Press Ctrl+C when done.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"Your chat ID is: {chat_id}\n\n"
        f"Add this to your .env file:\n"
        f"TELEGRAM_CHAT_ID={chat_id}\n\n"
        f"Then press Ctrl+C to stop this script."
    )
    print(f"\nChat ID found: {chat_id}")
    print(f"Add to .env:  TELEGRAM_CHAT_ID={chat_id}")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    print("Bot is running. Send /start to your bot in Telegram...")
    print("Press Ctrl+C to stop.\n")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
