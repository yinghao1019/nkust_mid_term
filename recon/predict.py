# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
"""Sample prediction script for TensorFlow 2.x."""
import sys
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
#匯入自訂義模組
from recon.object_detection import ObjectDetection
from recon.img_processing import img_BoundingCut
MODEL_FILENAME = 'D:/acer/Desktop/mid_term_F108157110/recon/model.pb'
LABELS_FILENAME = 'D:/acer/Desktop/mid_term_F108157110/recon/labels.txt'


class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]


def predict_main(image):
    #create storage pic predict info
    predict_cutImage=[]
    # Load a TensorFlow model
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    print('model label{}'.format(labels))
    od_model = TFObjectDetection(graph_def, labels)
    print('model{}'.format(od_model))
    img_cutArr=img_BoundingCut(image)
    for i,img_cut in enumerate(img_cutArr):
        #write pic &use PIL to read
        cv2.imwrite('image_{}.jpg'.format(i),img_cut)
        image=Image.open('image_{}.jpg'.format(i))
        predictions = od_model.predict_image(image)
        print("predict {}th's pic".format(i))
        h,w=img_cut.shape
        #顯示圖片標記的部分
        ret=predictions[0]
        prob=ret['probability']
        tagname=ret['tagName']
        tagId=ret['tagId']
        bbox=ret['boundingBox']
        left=bbox['left']
        top=bbox['top']
        width=bbox['width']
        height=bbox['height']
        #compute pic locate x,y
        x1=int(left*w)
        y1=int(top*h)
        x2=int(x1+width*w)
        y2=int(y1+height*h)
        print('the {}tag{} locate{x1},{y1},{x2},{y2}'.format(tagId,tagname,x1=x1,y1=y1,x2=x2,y2=y2))
        #put text of img
        img_str='tag{}:{:.2f}'.format(tagname,prob)
        p0=(max(x1,15),max(y1,15))
        #append model predictions in image
        predict_cutImage.append((i,tagname))
    return predict_cutImage