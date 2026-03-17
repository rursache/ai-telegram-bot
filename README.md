# AI Telegram Bot

A Telegram bot that supports both **OpenAI** and **Anthropic** APIs with streaming responses, vision support, and group chat integration.

## Features

- **Multi-provider**: Switch between OpenAI and Anthropic with a single env variable
- **Streaming responses**: Real-time message updates as the AI generates text
- **Vision**: Send images and get AI-powered descriptions
- **Group chats**: Trigger keyword support, reply detection
- **Inline queries**: Use the bot in any chat via `@botusername`
- **Conversation history**: Automatic summarization when history gets too long
- **Access control**: Restrict usage to specific Telegram user IDs

## Quick Start

### Docker (recommended)

Edit the `docker-compose.yml` with your configuration and run:

```bash
docker compose up -d
```

The compose file uses the pre-built image from GHCR and supports all configuration via environment variables inline. See `docker-compose.yml` for a full example with all available options.

You can also use a `.env` file instead of inline environment variables.

### Run directly

```bash
pip install -r requirements.txt
python bot/main.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVIDER` | `openai` | AI provider: `openai` or `anthropic` |
| `API_KEY` | - | API key for your chosen provider |
| `TELEGRAM_BOT_TOKEN` | - | Telegram bot token from [@BotFather](https://t.me/BotFather) |
| `MODEL` | auto | Model name (defaults based on provider) |
| `ADMIN_USER_IDS` | `-` | Comma-separated admin Telegram user IDs |
| `ALLOWED_TELEGRAM_USER_IDS` | `*` | Comma-separated allowed user IDs, or `*` for all |
| `ASSISTANT_PROMPT` | `You are a helpful assistant.` | System prompt |
| `MAX_TOKENS` | `16384` | Max completion tokens |
| `MAX_HISTORY_SIZE` | `15` | Max messages in conversation history |
| `MAX_CONVERSATION_AGE_MINUTES` | `180` | Auto-reset conversation after inactivity |
| `ENABLE_VISION` | `true` | Enable image interpretation |
| `ENABLE_QUOTING` | `true` | Quote the original message in replies |
| `GROUP_TRIGGER_KEYWORD` | - | Keyword to trigger bot in group chats |
| `STREAM` | `true` | Enable streaming responses |

## Commands

- `/reset` - Reset conversation (optionally with a new system prompt)
- `/resend` - Resend last message
- `/chat` - Chat in group conversations

## License

This project is licensed under the GPL-2.0 License.
