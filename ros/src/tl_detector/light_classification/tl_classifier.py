from styx_msgs.msg import TrafficLight

import os
import cv2
import numpy as np
import rospy
import tensorflow as tf

#Helper Functions in generating graph
def load_graph(model_path):
	"""
	Function to load the frozen inference graph

	Input: File path to frozen model file
 
	Output: Graph
	"""

	graph = tf.Graph()
	with graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(model_path,'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	return graph

def filter_results(min_score, scores, classes):
    """
	Returns tuple (scores, classes) for all scores above min_score

	Input: scores for each classe detected and the minimum score threshold

	Ouput: Tuple of filtered (scores, classes)`

    """

    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)

    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    
    return filtered_scores, filtered_classes



class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        
        self.light_detected = TrafficLight.UNKNOWN

        #Hard Coding the model file path 
        #print(os.getcwd())
        #self.model_path = 'light_classification/models/frozen_inference_graph.pb'
        self.model_path = 'light_classification/models/frozen_inference_graph_1.pb'

        #Load Graph with the help of Helper Function
        self.detection_graph = load_graph(self.model_path)

        #Defining Input and Output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        #Tensorflow session for detection
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph = self.detection_graph, config = self.config)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        (scores, classes) = self.sess.run([self.detection_scores, self.detection_classes], 
                                        		feed_dict={self.image_tensor: image_np})

    	scores = np.squeeze(scores)
    	classes = np.squeeze(classes)

    	confidence_cutoff = 0.5
    	# Filter boxes with a confidence score less than `confidence_cutoff`
    	scores, classes = filter_results(confidence_cutoff, scores, classes)

    	if(len(scores)>1):
    		return classes[0]-1
    	else:
    		return TrafficLight.UNKNOWN 

        return TrafficLight.UNKNOWN 
