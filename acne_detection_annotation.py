from flask import Flask, flash, request, url_for, render_template, redirect
import os
from os.path import join,splitext
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras import backend as K
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
import shutil
import cv2
import imageio
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils
from imutils import paths
import imutils


ALLOWED_EXTENSION  =set(['png','jpg','jpeg'])
COCO_TRAINED_MODEL = 'models/mask_rcnn_.1601370701.9844453.h5'
UPLOAD_FOLDER = 'static/uploads/'

# initialize the class names dictionary
CLASS_NAMES = {1: "acne"}


class AcneConfig(Config):
    # give the configuration a recognizable name
    NAME = "acne"

    # set the number of GPUs to use training along with the number of
    # images per GPU (which may have to be tuned depending on how
    # much memory your GPU has)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # set the number of steps per training epoch
    STEPS_PER_EPOCH = 42 #len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)
    
    # number of classes (+1 for the background)
    NUM_CLASSES = 2
    

class AcneInferenceConfig(AcneConfig):
    # set the number of GPUs and images per GPU (which may be
    # different values than the ones used for training)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # set the minimum detection confidence (used to prune out false
    # positive detections)
    DETECTION_MIN_CONFIDENCE = 0.9

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def allowed_file(filename):
    return '.' in filename and \
     filename.rsplit('.',1)[1] in ALLOWED_EXTENSION

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('ImageML.html')

@app.route('/api/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No posted image. Should be attribute named image')
    file = request.files['image']
    
    if file.filename =='':
        return render_template('ImageML.html', prediction = 'You did not select an image')
    
    if file and allowed_file(file.filename):
        items = []
        
        use_tf_keras = False
        
        filename = file.filename
        originalFile = splitext(os.path.basename(filename))[0] + "_original_image" + splitext(os.path.basename(filename))[1]
        input_path = UPLOAD_FOLDER
        print("File Name --> "+filename)
        split_filename = filename.split(".")[0]
        split_filename = split_filename.split("_")[0]
        print("Image prefix value --> "+split_filename)

        if os.path.exists(input_path) and os.path.isdir(input_path):
            shutil.rmtree(input_path)
            print('input directory removed')

        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        
        K.clear_session()
        config = AcneInferenceConfig()
        
        model = modellib.MaskRCNN(mode="inference", config=config,
            model_dir='./')
        
        
        
        model.load_weights(COCO_TRAINED_MODEL, by_name=True)
        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        actual_image = image.copy()
        actual_image = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        actual_image = cv2.cvtColor(np.asarray(actual_image), cv2.COLOR_RGB2BGR)
        actual_image = imutils.resize(actual_image, width=600)
        image = imutils.resize(image, width=600)

        r = model.detect([image], verbose=1)[0]
        

        for i in range(0, r["rois"].shape[0]):
            mask = r["masks"][:, :, i]
            image = visualize.apply_mask(image, mask,
                (1.0, 0.0, 0.0), alpha=0.5)
            image = visualize.draw_box(image, r["rois"][i],
                (1.0, 0.0, 0.0))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for i in range(0, len(r["scores"])):
            (startY, startX, endY, end) = r["rois"][i]
            classID = r["class_ids"][i]
            label = CLASS_NAMES[classID]
            score = r["scores"][i]

            text = "{}: {:.4f}".format(label, score)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if not os.path.exists(input_path):
            os.makedirs(input_path)
            imageio.imwrite(input_path + filename, image)
            imageio.imwrite(input_path + originalFile, actual_image)
        
        #time.sleep(5)
        if len(r["scores"]) > 1:
            items.append('Acne Detected')
        else:
            items.append('No Acne Detected')

        response = {'Prediction': items}

        basepath = 'static/uploads'
        imagespath = os.path.join(basepath,'*g')
        images_list = {}

        for imag in glob.glob(imagespath):
            if "_original_image" in splitext(os.path.basename(imag))[0]:
                images_list[0] = originalFile
            else:
                images_list[1] = filename
        
        flash('Image successfully uploaded and displayed : {}'.format(response))
        return render_template('ImageML.html', filename = images_list[1], filename1 = images_list[0])
    else:
        return render_template('ImageML.html', prediction = 'Invalid File extension')

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)