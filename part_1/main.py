import cv2
import numpy as np


TEXT = "Hello world, it's me Mariï*iòó0"
IMG_PATH = "tree.jpg"
NEW_PATH = "tree_new.png"
NB_BITS = 8


def standerdize_length(caracter):
    if len(caracter) < NB_BITS:
        return ['0'] * (NB_BITS - len(caracter)) + caracter
    return caracter


def to_bin(n):
    # convert and remove 0b at the begining
    return list(bin(n)[2:])


def change_last_bit(value, bit):
    value = to_bin(value)
    value[-1] = bit
    return int("".join(value), 2)
    

def write(text, img_path, new_path):

    img = cv2.imread(img_path)

    h, v, c = img.shape

    if img is None: print("image empty")

    if len(text)+1 > (h*v*c)/NB_BITS : raise ValueError()

    else:

        text_bit = []
        for char in text:
            text_bit += standerdize_length(to_bin(ord(char)))

        text_bit = np.array(text_bit + ['0'] * NB_BITS)

        img_flat = img.ravel()

        for i, p in enumerate(text_bit):
            img_flat[i] = change_last_bit(img_flat[i], p)

        cv2.imwrite(new_path, img)
    return img


def extract_text(bits):

    low, high = 0, NB_BITS
    string = ''
    length = len(bits)

    while high <= length:

        string += chr(int("".join(bits[low:high]), 2))

        low += NB_BITS
        high += NB_BITS

    return string


def read(img_path):

    img = cv2.imread(img_path)

    if img is None : print("image empty")

    else:

        last_bit = lambda x: '0' if x % 2 == 0 else '1'

        nb_zeros = 0
        bits = []

        for i, num in enumerate(img.reshape(-1)):

            l_bit = last_bit(num)

            bits.append(l_bit)

            if l_bit == '0' : nb_zeros += 1

            if (i+1) % NB_BITS == 0:
                if nb_zeros == NB_BITS: break
                else : nb_zeros = 0

        bits = bits[:-NB_BITS]

        return extract_text(bits)


if __name__ == '__main__':

    write(TEXT, IMG_PATH, NEW_PATH)
    string = read(NEW_PATH)
    print(string)