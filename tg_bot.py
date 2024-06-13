import json
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, InputMediaPhoto
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
from PIL import Image
from library.insert_everything import InsertEvetything


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define states
CHOOSE_LOCATION, CHOOSE_COUNT, UPLOAD_IMAGE = range(3)

# Thread pool for blocking calls
executor = ThreadPoolExecutor()


# Temporary function
def temp_func(img: Image, results_count: int, progress_callback, generation_location="all"):
    test_img = "test_imgs/handled_chair.webp"
    locations_count = 20

    for i in range(locations_count):
        time.sleep(1)
        progress_callback(i + 1, locations_count)

    return [Image.open(test_img) for _ in range(results_count)]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("Внутри дома", callback_data="indoor")],
        [InlineKeyboardButton("Вне дома", callback_data="outdoor")],
        [InlineKeyboardButton("Определить автоматически", callback_data="automatic")],
        [InlineKeyboardButton("Подойдет любой вариант", callback_data="all")],
        [InlineKeyboardButton("Отмена", callback_data="cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите аргумент пайплайна:", reply_markup=reply_markup)
    return CHOOSE_LOCATION


async def choose_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "cancel":
        return await cancel(update, context)
    context.user_data["location"] = query.data
    keyboard = [
        [InlineKeyboardButton("Отмена", callback_data="cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(text=f"Вы выбрали: {query.data}. Сколько картинок вы хотите увидеть? (Максимум 20)",
                                  reply_markup=reply_markup)
    return CHOOSE_COUNT


async def choose_count(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    count = int(update.message.text)
    if count > 20:
        await update.message.reply_text("Пожалуйста, выберите число не больше 20.")
        return CHOOSE_COUNT
    context.user_data["count"] = count
    keyboard = [
        [InlineKeyboardButton("Отмена", callback_data="cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Загрузите картинку:", reply_markup=reply_markup)
    return UPLOAD_IMAGE


async def upload_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    await photo_file.download_to_drive("temp_image.jpg")
    img = Image.open("temp_image.jpg")
    results_count = context.user_data["count"]
    generation_location = context.user_data["location"]

    progress_message = await update.message.reply_text("Начинаю обработку...")

    def progress_callback(progress, total):
        asyncio.run_coroutine_threadsafe(
            progress_message.edit_text(f"Прогресс: {progress}/{total}"), context.application.loop
        )

    loop = asyncio.get_event_loop()
    result_images = await loop.run_in_executor(
        executor, PIPELINE, img, results_count, generation_location, progress_callback
    )

    # Сохраняем изображения и подготавливаем их для отправки
    media = []
    for i, image in enumerate(result_images):
        file_path = f"result_image_{i}.jpg"
        image.save(file_path)
        media.append(InputMediaPhoto(open(file_path, "rb")))

    await update.message.reply_media_group(media)

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Отменено. Вы можете начать заново, отправив /start.")
    return ConversationHandler.END


def main() -> None:
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSE_LOCATION: [CallbackQueryHandler(choose_location)],
            CHOOSE_COUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_count)],
            UPLOAD_IMAGE: [MessageHandler(filters.PHOTO, upload_image)],
        },
        fallbacks=[CallbackQueryHandler(cancel, pattern='cancel')],
    )

    application.add_handler(conv_handler)

    application.run_polling()


def build_pipeline():
    with open('data.json', 'r') as f:
        data = json.load(f)
    return InsertEvetything(data)


PIPELINE = build_pipeline()

if __name__ == "__main__":
    main()
