# Import required modules
import numpy as np
import cv2 as cv
import math
import argparse

############ Add argument parser for command line arguments ############
parser = argparse.ArgumentParser(
    description = "The OCR model can be obtained from converting the pretrained CRNN model to .onnx format from the github repository https://github.com/meijieru/crnn.pytorch")
parser.add_argument('--input',
                    help='Path to input image. Skip this argument to capture frames from a camera.')
parser.add_argument('--ocr', default="/home/moo/Desktop/ocr/clovaai/new_model_8_4/DenseNet.onnx",
                    help="Path to a binary .pb or .onnx file contains trained recognition network", )
parser.add_argument('--width', type=int, default=100,
                    help='Preprocess input image by resizing to a specific width.')
parser.add_argument('--height', type=int, default=32,
                    help='Preprocess input image by resizing to a specific height.')
args = parser.parse_args()


############ Utility functions ############

def fill_img(img, width = 100, height = 32):
    # print(img.shape)
    h, w = img.shape[0], img.shape[1]
    ratio = w/float(h)
    resizedW = width
    if math.ceil( height * ratio ) > width:
        resizedW = width
    else:
        resizedW = math.ceil( height * ratio )
    
    resizedImg = cv.resize(img, (resizedW, height))

    # print(resizedImg.shape)
    if resizedW != width:
        repetition = [1] * (resizedW-1)
        repetition.append(width - resizedW)
        print(len(repetition))
        resizedImg = np.repeat(resizedImg, repetition, axis= 1)
    
    return resizedImg


def decodeText(scores):
    text = ""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        print(c)
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'

    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)


def main():
    # Read and store arguments
    modelRecognition = args.ocr
    imagePath = args.input
    inpWidth = args.width
    inpHeight = args.height
    # Load network
    recognizer = cv.dnn.readNetFromONNX(modelRecognition)
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("name", img)
    cv.waitKey(100)

    # if use padding
    # img = fill_img(img, inpWidth, inpHeight)

    blob = cv.dnn.blobFromImage(img, size=(inpWidth, inpHeight), mean=127.5, scalefactor=1 / 255.0)
    blob -= 0.5
    blob /= 0.5
    recognizer.setInput(blob)

    # Run the recognition model
    result = recognizer.forward()

    # decode the result into text
    wordRecognized = decodeText(result)
    print("recog output is : ", wordRecognized)


if __name__ == "__main__":
    main()
