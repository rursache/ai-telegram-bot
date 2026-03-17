from __future__ import annotations

import asyncio
import io
import logging

from uuid import uuid4
from telegram import Update, constants, BotCommand, BotCommandScopeAllGroupChats
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle, InputTextMessageContent
from telegram.error import RetryAfter, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.ext import InlineQueryHandler, CallbackQueryHandler, Application, ContextTypes, CallbackContext

from ai_helper import AIHelper


class AITelegramBot:
    def __init__(self, config: dict, ai: AIHelper):
        self.config = config
        self.ai = ai
        self.last_message: dict[int, str] = {}
        self.inline_queries_cache: dict[str, str] = {}

        self.commands = [
            BotCommand(command='reset', description='Reset conversation'),
            BotCommand(command='resend', description='Resend last message'),
        ]
        self.group_commands = [
            BotCommand(command='chat', description='Chat in groups'),
        ] + self.commands
        self.unsupported_message = "This file type is not supported. I can only process text messages and images."

    # -- Handlers --

    async def reset(self, update: Update, _: ContextTypes.DEFAULT_TYPE):
        if not await self._is_allowed(update):
            return

        chat_id = update.effective_chat.id
        reset_content = _message_text(update.message)
        self.ai.reset_chat_history(chat_id=chat_id, content=reset_content)

        logging.info(f'Reset conversation for {update.message.from_user.name} (id: {update.message.from_user.id})')
        await update.effective_message.reply_text(
            message_thread_id=_thread_id(update),
            text="Conversation reset."
        )

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_allowed(update):
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            await update.effective_message.reply_text(
                message_thread_id=_thread_id(update),
                text="No previous message to resend."
            )
            return

        logging.info(f'Resending last prompt from {update.message.from_user.name}')
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)
        await self.prompt(update=update, context=context)

    async def vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.config['enable_vision'] or not await self._is_allowed(update):
            return

        chat_id = update.effective_chat.id
        prompt = update.message.caption

        if _is_group(update):
            if self.config['ignore_group_vision']:
                return
            trigger = self.config['group_trigger_keyword']
            if (prompt is None and trigger != '') or \
                    (prompt is not None and not prompt.lower().startswith(trigger.lower())):
                return

        image = update.message.effective_attachment[-1]

        async def _execute():
            try:
                media_file = await context.bot.get_file(image.file_id)
                temp_file = io.BytesIO(await media_file.download_as_bytearray())
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=_thread_id(update),
                    text=f"Failed to download image: {e}"
                )
                return

            await self._stream_to_chat(update, context, chat_id,
                                       self.ai.interpret_image(chat_id=chat_id, fileobj=temp_file, prompt=prompt))

        await _with_typing(update, context, _execute)

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.edited_message or not update.message or update.message.via_bot:
            return
        if not await self._is_allowed(update):
            return

        logging.info(f'New message from {update.message.from_user.name} (id: {update.message.from_user.id})')
        chat_id = update.effective_chat.id
        prompt = _message_text(update.message)
        self.last_message[chat_id] = prompt

        if _is_group(update):
            trigger = self.config['group_trigger_keyword']
            if prompt.lower().startswith(trigger.lower()) or update.message.text.lower().startswith('/chat'):
                if prompt.lower().startswith(trigger.lower()):
                    prompt = prompt[len(trigger):].strip()
                if update.message.reply_to_message and \
                        update.message.reply_to_message.text and \
                        update.message.reply_to_message.from_user.id != context.bot.id:
                    prompt = f'"{update.message.reply_to_message.text}" {prompt}'
            elif update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                logging.info('Reply to bot in group, allowing...')
            else:
                return

        try:
            await update.effective_message.reply_chat_action(
                action=constants.ChatAction.TYPING,
                message_thread_id=_thread_id(update),
            )
            stream = self.ai.get_chat_response(chat_id=chat_id, query=prompt)
            await self._stream_to_chat(update, context, chat_id, stream)
        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=_thread_id(update),
                reply_to_message_id=_reply_id(self.config, update),
                text=f"Error: {e}",
            )

    async def unsupported(self, update: Update, _: ContextTypes.DEFAULT_TYPE):
        if not await self._is_allowed(update):
            return
        await update.effective_message.reply_text(
            message_thread_id=_thread_id(update),
            text=self.unsupported_message,
        )

    async def inline_query(self, update: Update, _: ContextTypes.DEFAULT_TYPE):
        query = update.inline_query.query
        if len(query) < 3:
            return
        if not self._is_user_allowed(update.inline_query.from_user.id):
            return

        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query

        try:
            reply_markup = InlineKeyboardMarkup([[
                InlineKeyboardButton(text='🤖 Get answer', callback_data=f'q:{result_id}')
            ]])
            await update.inline_query.answer([
                InlineQueryResultArticle(
                    id=result_id,
                    title='Ask AI',
                    input_message_content=InputTextMessageContent(query),
                    description=query,
                    reply_markup=reply_markup,
                )
            ], cache_time=0)
        except Exception as e:
            logging.error(f'Inline query error: {e}')

    async def handle_callback_inline_query(self, update: Update, context: CallbackContext):
        callback_data = update.callback_query.data
        user_id = update.callback_query.from_user.id
        inline_message_id = update.callback_query.inline_message_id

        if not callback_data.startswith('q:'):
            return

        unique_id = callback_data.split(':')[1]
        query = self.inline_queries_cache.pop(unique_id, None)
        if not query:
            await _edit_message(context, chat_id=None, message_id=inline_message_id,
                                text="Query expired. Please try again.", is_inline=True)
            return

        try:
            stream = self.ai.get_chat_response(chat_id=user_id, query=query)
            i = 0
            prev = ''
            backoff = 0
            async for content, tokens in stream:
                if not content.strip():
                    continue

                cutoff = _stream_cutoff(update, content) + backoff
                if i == 0:
                    try:
                        await _edit_message(context, chat_id=None, message_id=inline_message_id,
                                            text=f'{query}\n\nAnswer:\n{content}', is_inline=True)
                    except Exception:
                        continue
                elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                    prev = content
                    use_md = tokens != 'not_finished'
                    text = f'{query}\n\n{"_" if use_md else ""}Answer:{"_" if use_md else ""}\n{content}'
                    try:
                        await _edit_message(context, chat_id=None, message_id=inline_message_id,
                                            text=text[:4096], markdown=use_md, is_inline=True)
                    except RetryAfter as e:
                        backoff += 5
                        await asyncio.sleep(e.retry_after)
                    except (TimedOut, Exception):
                        backoff += 5
                    await asyncio.sleep(0.01)
                i += 1

        except Exception as e:
            logging.exception(e)
            await _edit_message(context, chat_id=None, message_id=inline_message_id,
                                text=f"{query}\n\nError: {e}", is_inline=True)

    # -- Streaming helper --

    async def _stream_to_chat(self, update: Update, context, chat_id: int, stream):
        """Stream AI response to a Telegram chat, editing message as content arrives."""
        i = 0
        prev = ''
        sent_message = None
        backoff = 0
        stream_chunk = 0

        async for content, tokens in stream:
            if not content.strip():
                continue

            chunks = _split_chunks(content)
            if len(chunks) > 1:
                content = chunks[-1]
                if stream_chunk != len(chunks) - 1:
                    stream_chunk += 1
                    try:
                        await _edit_message(context, chat_id, str(sent_message.message_id), chunks[-2])
                    except Exception:
                        pass
                    try:
                        sent_message = await update.effective_message.reply_text(
                            message_thread_id=_thread_id(update),
                            text=content if content else "...",
                        )
                    except Exception:
                        pass
                    continue

            cutoff = _stream_cutoff(update, content) + backoff

            if i == 0:
                try:
                    if sent_message:
                        await context.bot.delete_message(
                            chat_id=sent_message.chat_id, message_id=sent_message.message_id)
                    sent_message = await update.effective_message.reply_text(
                        message_thread_id=_thread_id(update),
                        reply_to_message_id=_reply_id(self.config, update),
                        text=content,
                    )
                except Exception:
                    continue
            elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                prev = content
                try:
                    use_md = tokens != 'not_finished'
                    await _edit_message(context, chat_id, str(sent_message.message_id),
                                        text=content, markdown=use_md)
                except RetryAfter as e:
                    backoff += 5
                    await asyncio.sleep(e.retry_after)
                    continue
                except TimedOut:
                    backoff += 5
                    await asyncio.sleep(0.5)
                    continue
                except Exception:
                    backoff += 5
                    continue
                await asyncio.sleep(0.01)

            i += 1

    # -- Auth --

    async def _is_allowed(self, update: Update) -> bool:
        user_id = update.message.from_user.id
        if not self._is_user_allowed(user_id):
            logging.warning(f'User {update.message.from_user.name} (id: {user_id}) not allowed')
            await update.effective_message.reply_text(
                message_thread_id=_thread_id(update),
                text="You are not allowed to use this bot.",
            )
            return False
        return True

    def _is_user_allowed(self, user_id: int) -> bool:
        if self.config['allowed_user_ids'] == '*':
            return True
        allowed = self.config['allowed_user_ids'].split(',')
        admin = self.config['admin_user_ids'].split(',')
        return str(user_id) in allowed or str(user_id) in admin

    # -- Bot setup --

    async def _post_init(self, application: Application):
        await application.bot.set_my_commands(self.group_commands, scope=BotCommandScopeAllGroupChats())
        await application.bot.set_my_commands(self.commands)

    def run(self):
        application = ApplicationBuilder() \
            .token(self.config['token']) \
            .post_init(self._post_init) \
            .concurrent_updates(True) \
            .build()

        application.add_handler(CommandHandler('reset', self.reset))
        application.add_handler(CommandHandler('resend', self.resend))
        application.add_handler(CommandHandler(
            'chat', self.prompt, filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP))
        application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self.vision))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))
        application.add_handler(MessageHandler(
            filters.ATTACHMENT & ~filters.PHOTO & ~filters.Document.IMAGE, self.unsupported))
        application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[
            constants.ChatType.GROUP, constants.ChatType.SUPERGROUP, constants.ChatType.PRIVATE
        ]))
        application.add_handler(CallbackQueryHandler(self.handle_callback_inline_query))
        application.add_error_handler(_error_handler)
        application.run_polling()


# -- Utility functions --

def _message_text(message) -> str:
    from telegram import MessageEntity
    text = message.text or ''
    for _, cmd_text in sorted(message.parse_entities([MessageEntity.BOT_COMMAND]).items(),
                              key=lambda item: item[0].offset):
        text = text.replace(cmd_text, '').strip()
    return text


def _thread_id(update: Update) -> int | None:
    if update.effective_message and update.effective_message.is_topic_message:
        return update.effective_message.message_thread_id
    return None


def _is_group(update: Update) -> bool:
    if not update.effective_chat:
        return False
    return update.effective_chat.type in [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]


def _reply_id(config, update: Update):
    if config['enable_quoting'] or _is_group(update):
        return update.message.message_id
    return None


def _split_chunks(text: str, size: int = 4096) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def _stream_cutoff(update: Update, content: str) -> int:
    if _is_group(update):
        return 180 if len(content) > 1000 else 120 if len(content) > 200 else 90 if len(content) > 50 else 50
    return 90 if len(content) > 1000 else 45 if len(content) > 200 else 25 if len(content) > 50 else 15


async def _edit_message(context: ContextTypes.DEFAULT_TYPE, chat_id, message_id: str,
                        text: str, markdown: bool = True, is_inline: bool = False):
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(message_id) if not is_inline else None,
            inline_message_id=message_id if is_inline else None,
            text=text,
            parse_mode=constants.ParseMode.MARKDOWN if markdown else None,
        )
    except Exception as e:
        if "Message is not modified" in str(e):
            return
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id) if not is_inline else None,
                inline_message_id=message_id if is_inline else None,
                text=text,
            )
        except Exception:
            raise


async def _with_typing(update: Update, context, coroutine):
    task = context.application.create_task(coroutine(), update=update)
    while not task.done():
        context.application.create_task(
            update.effective_chat.send_action(constants.ChatAction.TYPING, message_thread_id=_thread_id(update))
        )
        try:
            await asyncio.wait_for(asyncio.shield(task), 4.5)
        except asyncio.TimeoutError:
            pass


async def _error_handler(_: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f'Exception while handling an update: {context.error}')
