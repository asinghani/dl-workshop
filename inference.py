import tensorflow as tf
import numpy as np
import cv2

detection_graph = tf.Graph()
with detection_graph.as_default():
    gd = tf.GraphDef()
    with tf.gfile.GFile("/home/ubuntu/object-detection-model.pb", "rb") as file:
        gd.ParseFromString(file.read())
        tf.import_graph_def(gd, name="")

def infer_image(image, graph, image_tensor):
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

    output_dict["num_detections"] = int(output_dict["num_detections"][0])
    output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(np.uint8)
    output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
    output_dict["detection_scores"] = output_dict["detection_scores"][0]

    return output_dict

cap = cv2.VideoCapture(1)

with detection_graph.as_default():
    with tf.Session() as sess:
        ops = tf.get_default_graph().get_operations()
        all_tensors = {output.name for op in ops for output in op.outputs}

        tensor_dict = {}
        for key in ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]:
            tensor_name = key + ":0"

            if tensor_name in all_tensors:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

        while True:
            image = cap.read()[1]
            h, w = image.shape[0:2]
            image_scaled = image[:, :, ::-1] #/ 255.0

            out = infer_image(image_scaled, detection_graph, image_tensor)
            print(out)

            for i in range(len(out["detection_scores"])):
                if out["detection_scores"][i] > 0.6:
                    if out["detection_classes"][i] == 1: 
                        ymin, xmin, ymax, xmax = out["detection_boxes"][i]

                        cv2.rectangle(image, (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h)), (0, 255, 0), 3)

            cv2.imshow("asdkjahsdkads", image)
            cv2.waitKey(5)

