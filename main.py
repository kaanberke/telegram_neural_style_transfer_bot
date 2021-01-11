import logging
from io import BytesIO
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import PIL




def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def process_tensor(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Hi!")


def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Help!")


def echo(update: Update, context: CallbackContext):
    update.message.reply_text(update.message.text)


def photo(update: Update, context: CallbackContext):
    global image_no

    file = context.bot.get_file(update.message.photo[-1].file_id)
    img_path = os.path.join("images", f"{image_no}.jpg")
    stylized_img_path = os.path.join("stylized_images", f"{image_no}.jpg")
    file.download(img_path)

    content_image = load_img(img_path)
    style_image = load_img(style_path)

    tensor = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    tensor = process_tensor(tensor)

    stylized_image = PIL.Image.fromarray(tensor)
    stylized_image.save(stylized_img_path)

    with open(stylized_img_path, "rb") as f:
        img = BytesIO(f.read())
    context.bot.send_photo(chat_id=update.message.chat_id, photo=img)
    image_no += 1


def main():
    if not os.path.isdir("images"):
        os.mkdir("images")

    if not os.path.isdir("stylized_images"):
        os.mkdir("stylized_images")

    updater = Updater("TOKEN", use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))
    dispatcher.add_handler(MessageHandler(Filters.photo, photo))
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    style_path = tf.keras.utils.get_file('kandinsky5.jpg',
                                         'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    image_no = 0
    main()
