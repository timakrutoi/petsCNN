#!/usr/bin/env python


import logging

import telegram
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

from main import predict
import numpy as np

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Try command /Breeds')


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def photo_handler(update, context):
    """Download photo and call predict()"""
    doc = update.message['photo'][-1].get_file()
    file_name = update.message['photo'][-1].get_file()['file_path'].split(sep='/')[-1]

    pic_name = f'saved_from_tg/{file_name}'
    context.bot.get_file(doc).download(pic_name)

    update.message.reply_text('File successfully downloaded')

    update.message.reply_text(final_predict(pic_name))


def file_handler(update, context):
    """Download file and call predict()"""
    fileName = update.message['document']['file_name']
    pic_name = f'saved_from_tg/{fileName}'
    context.bot.get_file(update.message.document).download(pic_name)
    update.message.reply_text('File successfully downloaded')

    update.message.reply_text(final_predict(pic_name))


def final_predict(pic_name, model_name='petsCNN.onnx'):
    # todo say its bad prediction if its bad

    if not pic_name.endswith('.jpg'):
        return 'Sorry. Its not a valid format. Use jpg.'

    labels = {
        '0': 'Maine_Coon',
        '1': 'leonberger',
        '2': 'pug',
        '3': 'Bombay',
        '4': 'beagle',
        '5': 'keeshond',
        '6': 'havanese',
        '7': 'newfoundland',
        '8': 'scottish_terrier',
        '9': 'Abyssinian',
        '10': 'american_bulldog',
        '11': 'Siamese',
        '12': 'saint_bernard',
        '13': 'german_shorthaired',
        '14': 'shiba_inu',
        '15': 'samoyed',
        '16': 'Sphynx',
        '17': 'staffordshire_bull_terrier',
        '18': 'chihuahua',
        '19': 'great_pyrenees',
        '20': 'Bengal',
        '21': 'Russian_Blue',
        '22': 'basset_hound',
        '23': 'english_setter',
        '24': 'Persian',
        '25': 'american_pit_bull_terrier',
        '26': 'yorkshire_terrier',
        '27': 'japanese_chin',
        '28': 'Birman',
        '29': 'Egyptian_Mau',
        '30': 'British_Shorthair',
        '31': 'boxer',
        '32': 'wheaten_terrier',
        '33': 'pomeranian',
        '34': 'Ragdoll',
        '35': 'english_cocker_spaniel',
        '36': 'miniature_pinscher'
    }

    res = predict(pic_name, model_name)
    l = np.argsort(res)
    l = np.flip(l)

    out = ''

    if res[0, l[0,0]] > 10:
        out += 'petsCNN thinks its {}\n'.format(labels[f'{l[0, 0]}'])
        out += 'other likely predictions:\n'
        for i in range(1, 5):
            out += '\t\t\t({}) its {}\n'.format(i, labels[f'{l[0, i]}'])
    else:
        out += 'petsCNN is not sure about this, try another photo pls'

    return out


def breed_list_command(update: Update, context: CallbackContext):

    labels = {
        '0': 'Maine_Coon',
        '1': 'leonberger',
        '2': 'pug',
        '3': 'Bombay',
        '4': 'beagle',
        '5': 'keeshond',
        '6': 'havanese',
        '7': 'newfoundland',
        '8': 'scottish_terrier',
        '9': 'Abyssinian',
        '10': 'american_bulldog',
        '11': 'Siamese',
        '12': 'saint_bernard',
        '13': 'german_shorthaired',
        '14': 'shiba_inu',
        '15': 'samoyed',
        '16': 'Sphynx',
        '17': 'staffordshire_bull_terrier',
        '18': 'chihuahua',
        '19': 'great_pyrenees',
        '20': 'Bengal',
        '21': 'Russian_Blue',
        '22': 'basset_hound',
        '23': 'english_setter',
        '24': 'Persian',
        '25': 'american_pit_bull_terrier',
        '26': 'yorkshire_terrier',
        '27': 'japanese_chin',
        '28': 'Birman',
        '29': 'Egyptian_Mau',
        '30': 'British_Shorthair',
        '31': 'boxer',
        '32': 'wheaten_terrier',
        '33': 'pomeranian',
        '34': 'Ragdoll',
        '35': 'english_cocker_spaniel',
        '36': 'miniature_pinscher'
    }
    out = 'List of all available breeds:\n'

    for i, j in enumerate(labels.values()):
        out += f'{i+1}) {j}\n'

    update.message.reply_text(out)


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    f = open('token')
    TOKEN = f.read()[:-1]

    print(f'token is {TOKEN}')
    updater = Updater(TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('help', help_command))
    dispatcher.add_handler(CommandHandler(['Breeds', 'breeds'], breed_list_command))
    dispatcher.add_handler(MessageHandler(Filters.document, file_handler))
    dispatcher.add_handler(MessageHandler(Filters.photo, photo_handler))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
