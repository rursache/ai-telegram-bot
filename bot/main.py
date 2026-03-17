import logging
import os

from dotenv import load_dotenv

from ai_helper import create_ai_helper
from telegram_bot import AITelegramBot


def main():
    load_dotenv()

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    required_values = ['TELEGRAM_BOT_TOKEN', 'API_KEY']
    missing_values = [v for v in required_values if os.environ.get(v) is None]
    if missing_values:
        logging.error(f'Missing environment variables: {", ".join(missing_values)}')
        exit(1)

    provider = os.environ.get('PROVIDER', 'openai')

    ai_config = {
        'provider': provider,
        'api_key': os.environ['API_KEY'],
        'model': os.environ.get('MODEL', 'gpt-5-mini-2025-08-07' if provider == 'openai' else 'claude-sonnet-4-6'),
        'max_tokens': int(os.environ.get('MAX_TOKENS', 16384)),
        'max_history_size': int(os.environ.get('MAX_HISTORY_SIZE', 15)),
        'max_conversation_age_minutes': int(os.environ.get('MAX_CONVERSATION_AGE_MINUTES', 180)),
        'assistant_prompt': os.environ.get('ASSISTANT_PROMPT', 'You are a helpful assistant.'),
        'vision_prompt': os.environ.get('VISION_PROMPT', 'What is in this image?'),
    }

    telegram_config = {
        'token': os.environ['TELEGRAM_BOT_TOKEN'],
        'admin_user_ids': os.environ.get('ADMIN_USER_IDS', '-'),
        'allowed_user_ids': os.environ.get('ALLOWED_TELEGRAM_USER_IDS', '*'),
        'enable_quoting': os.environ.get('ENABLE_QUOTING', 'true').lower() == 'true',
        'enable_vision': os.environ.get('ENABLE_VISION', 'true').lower() == 'true',
        'ignore_group_vision': os.environ.get('IGNORE_GROUP_VISION', 'true').lower() == 'true',
        'group_trigger_keyword': os.environ.get('GROUP_TRIGGER_KEYWORD', ''),
        'stream': os.environ.get('STREAM', 'true').lower() == 'true',
    }

    ai_helper = create_ai_helper(ai_config)
    telegram_bot = AITelegramBot(config=telegram_config, ai=ai_helper)
    telegram_bot.run()


if __name__ == '__main__':
    main()
