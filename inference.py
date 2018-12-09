import tensorflow as tf
import numpy as np
import cv2

detection_graph = tf.Graph()
with detection_graph.as_default():
    gd = tf.GraphDef()
    with tf.gfile.GFile("~/model.pb", "rb") as file:
        gd.ParseFromString(file.read())
        tf.import_graph_def(gd, name="")

def infer_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensors = {output.name for op in ops for output in op.outputs}

            tensor_dict = {}
            for key in ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]:
                tensor_name = key + ":0"

                if tensor_name in all_tensors:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict["num_detections"] = int(output_dict["num_detections"][0])
            output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(np.uint8)
            output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
            output_dict["detection_scores"] = output_dict["detection_scores"][0]

    return output_dict

cap = cv2.VideoCapture(1)

while True:
    image = cap.read()[0]
    image_scaled = image / 255.0
    image_scaled = np.expand_dims(image, axis=0)

    out = infer_image(image_scaled, detection_graph)
    print(out)

