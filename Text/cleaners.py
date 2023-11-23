import re
from .mandarin import number_to_chinese, chinese_to_bopomofo, latin_to_bopomofo



def chinese_cleaners(text):
    '''Pipeline for Chinese text'''
    text = text.replace("[ZH]", "")
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = re.sub(r'([ˉˊˇˋ˙])$', r'\1。', text)
    return text
