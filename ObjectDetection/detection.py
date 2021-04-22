import os
import numpy as np 
import cv2


class TensorflowModel:
    def __init__(self, modelname, proto_filename, frozengraph_filename, classes_filename,
                base_confidence=0.6, classes_to_detect=["person", "car"]):

        """
        INPUTS:
            modelname(str)                   :Full name of the model in the "./model" directory
            proto_filename(str)              :Protofilename from the tensorflow for the model
            frozedgraph_filename(str)        :Frozen Tenosorflow graph  
            classes_filename(str)            :Filename for the class labels. The file should a .txt file and with each label name in newline. 
            base_confidence(float)           :The confidence level to accept the detection by the model.
                                              Eg. confidence of 0.6 means any detection which has detection prob more that 0.6 will be considered. 
            classes_to_detect(list[str, ..]) :List of classes that will be considered for the model. They should be in the label list.
        """
    
        # Model Parmaters
        self.BASE_CONFIDENCE   = base_confidence       # Base Confidence for the Object Detection
        self.CLASSES_TO_DETECT = classes_to_detect     # These are the only classes that will be considered for classification 
        self.modelname         = modelname             # Name of the model that we are loading

        # Load the Tensorflow Model
        self.__load_model(modelname, proto_filename, classes_filename, frozengraph_filename)

        # Sanity Checks
        self.__check_label()
        self.__check_confidence_level()
        
        
    #######################
    ## Utility functions ##
    #######################
    def __check_filepath(self, filepath):
        if not os.path.isfile(filepath):
            raise Exception("The {} does not exit!".format(os.path.basename(filepath)))

    def __check_label(self, ):
        if not all(item in self.CLASSES for item in self.CLASSES_TO_DETECT):
            raise Exception("Some classes_to_detect does not exit in the full list of classes loaded!")

    def __check_confidence_level(self, ):
        if (self.BASE_CONFIDENCE > 1.0) or (self.BASE_CONFIDENCE < 0.0):
            raise Exception("The base confidence level has to be in range (0.0, 1.0) !")

    ################
    ## Load Model ##
    ################
    def __load_model(self, model_filename, proto_filename, classes_filename, frozengraph_filename):        
        # MODEL Files paths
        model_directory = os.path.join(os.path.dirname(__file__), "models", model_filename)
        proto_filepath = os.path.join(model_directory, proto_filename)
        class_filepath = os.path.join(model_directory, classes_filename)
        graph_filepath = os.path.join(model_directory, frozengraph_filename)
        
        # First check if the files exits
        if not os.path.isdir(model_directory):
            raise Exception("Model dir does not exits! Please create a folder in ./model/{}".format(model_filename))
        self.__check_filepath(proto_filepath)
        self.__check_filepath(class_filepath)
        self.__check_filepath(graph_filepath)  
        
        # # Loading Up the Tensorflow Model # #
        print("[ObjD] Setting up the Tensorflow Model ... ", end=" ")
        # Load serialized model from disk
        def load(frozen_graph_path, proto_file):
            model = cv2.dnn.readNetFromTensorflow(frozen_graph_path, proto_file)
            return model
        self.MODEL = load(graph_filepath, proto_filepath)
        print("Done!")
        
        # Get the list of class labels from the Label file
        print("[ObjD] Reading Class Labels ...", end=" ")
        self.CLASSES = open(class_filepath).read().strip().split("\n")
        print("Done!")

    ################
    ## Detections ##
    ################
    def detect(self, image, imgName=None):
        """
        INPUT:
            image(numpy.ndarray)    :Numpy image array with 3-channels 
            imgName(str)            :Name of the image. Note: Only for printing purposes.
            
        RETURN:
            <list[dict]>
            Output the list of dictonary with {label, confidence, box} where,
            label       :Label of the object detected
            confidence  :Confidence level of the object detected
            bbox         :Bounding box of the object detected
        """

        # Return 
        objdetection = list()

        if imgName is not None:
            print("\n[ObjD] Detecting objects in " + str(imgName) + "...")
        else:
            print("\n[ObjD] Detecting objects ...")

        # Height and width of the image
        (height, width) = image.shape[:2]

        # Formating image for passing through the network for predictions
        print("Creating blob, ", end=" ")
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        self.MODEL.setInput(blob)
        print("Getting predictions, ", end=" ")
        detections = self.MODEL.forward()

        # After getting all the detections, gathreing label and bounding box.
        # based on base_confidence level and classes_to_detect
        detections = detections[0, 0]
        for detection in detections:
            # Get the confidence level in the detection
            confidence = float(detection[2])

            # If confidence level is greater than the base, then plot it on the image
            if confidence > self.BASE_CONFIDENCE:
                # Class of the detection
                class_id = int(detection[1])
                # Box dimensions for detection
                box = detection[3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # Check if the classes are in the considered group
                if self.CLASSES[class_id] in self.CLASSES_TO_DETECT:
                    # Add to the detection list 
                    objdetection.append({"label":self.CLASSES[class_id],
                                        "confidence": confidence * 100,
                                        "bbox":(startX, startY, endX, endY)})
        print("Done!")
        return objdetection



if __name__ == "__main__":
    pass

    