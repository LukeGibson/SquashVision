from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

def showImage(cv2Image, isA):
    global panelA, panelB, displayA, displayB

    # get frame dimension
    frameWidth = displayA.winfo_width()
    frameHeight = displayB.winfo_height()

    # calculate scales required to fit image in frame
    heightScale = int((frameHeight / cv2Image.shape[0]) * 100)
    widthScale = int((frameWidth / cv2Image.shape[1]) * 100)

    # select most restricting scale
    scale = heightScale
    if widthScale < heightScale:
        scale = widthScale

    # scale original image
    width = int(cv2Image.shape[1] * scale / 100)
    height = int(cv2Image.shape[0] * scale / 100)
    resizeDim = (width, height)
    image = cv2.resize(cv2Image, resizeDim, interpolation=cv2.INTER_AREA)

    # convert scale to tkinter format
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    if isA:
        # add image to frame A
        panelA.configure(image=image)
        panelA.image = image
    else:
        # add image to frame B
        panelB.configure(image=image)
        panelB.image = image


def selectFile():
    global currFile

    # open a file selection box
    filename = filedialog.askopenfilename(title="Select Image")
    currFile = filename

    if len(currFile) > 0:
        image = cv2.imread(currFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage(image, True)


def cannyEdge():
    global currFile
    print("Canny Edge Detection on: ", currFile)

    if len(currFile) > 0:
        # load image
        image = cv2.imread(currFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.Canny(image, 200, 250)

        showImage(image, False)


def backgroundSubtract():
    global currFile, displayB, panelB
    print("Background Subtraction on: ", currFile)


def operate():
    global currOp

    if currOp.get() == "CannyEdge Detection":
        cannyEdge()
    elif currOp.get() == "Background Subtraction":
        backgroundSubtract()


root = Tk()

# calculate relative app dimensions for screen resolution
screenHeight = root.winfo_screenheight()
screenWidth = root.winfo_screenwidth()

rHeight = round(screenHeight/1.2)
rWidth = round(screenWidth/1.8)
rDim = str(rWidth) + "x" + str(rHeight)

root.geometry(rDim)
root.title("Image Operators")

# store current file path and operation
currFile = ""
currOp = StringVar()
currOp.set("CannyEdge Detection")

# create container frames
displayA = Frame(root, bg="#cccccc")
displayA.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.01)

displayB = Frame(root, bg="#cccccc")
displayB.place(relwidth=0.75, relheight=0.48, relx=0.01, rely=0.51)

displayC = Frame(root)
displayC.place(relwidth=0.23, relheight=0.98, relx=0.76, rely=0.01)

# initialise labels to hold images
panelA = Label(displayA)
panelA.pack()
panelB = Label(displayB)
panelB.pack()

# create controls
openBut = Button(displayC, text="Open Image", padx=10, pady=5, command=selectFile)
openBut.pack(side="top", pady=0)

opSelect = OptionMenu(displayC, currOp, "CannyEdge Detection", "Background Subtraction")
opSelect.pack(side="top", pady=15)

operateBut = Button(displayC, text="Operate", padx=10, pady=5, command=operate)
operateBut.pack(side="top")

root.mainloop()
