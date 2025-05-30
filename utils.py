import cv2
import numpy as np


def smart_resize(img, new_size=512):
    """
    A very basic resizing function
    :param img: input image
    :param new_size: output max size
    :return: resized image (largest side = new_size), size ratio
    """
    ratio = new_size/max(img.shape[:2])
    return cv2.resize(img, None, fx=ratio, fy=ratio), ratio


def points_to_yolo(labels_df, points, part_id, img_h, img_w):
    """
    Create a contour from the list of X,Y points and get the bounding box
    :param labels_df: dataframe that will contain the boxes
    :param points: list of points (X,Y) of a facial landmark
    :param part_id: facial part ID
    :param img_h: image height
    :param img_w: image width
    :return: bounding box coordinates (not normalized)
    """
    contour = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    x_n, w_n = x / img_w, w / img_w
    y_n, h_n = y / img_h, h / img_h
    x_c = x_n + 0.5 * w_n
    y_c = y_n + 0.5 * h_n

    # Populating the dataframe
    labels_df.loc[len(labels_df), :] = [part_id, x_c, y_c, w_n, h_n]
    return x, y, w, h  # these are not the normalised coordinates, these are for plotting the box


def annotate_frame(image, detections, box_annotator, label_annotator, class_names_dict):
    """
    Annotate the bounding box with class name and confidence
    :param image: input image
    :param detections: YOLO detections object
    :param box_annotator: supervision bounding box annotator
    :param label_annotator: supervision bounding box annotator
    :param class_names_dict: dictionary with model's class names {class_id: class_name, ...}
    :return: annotated image
    """
    labels = [
        "{} {:0.2f}".format(class_names_dict[class_id], confidence)
        for _, _, confidence, class_id, _, _
        in detections
    ]
    image = box_annotator.annotate(scene=image, detections=detections)
    #image = label_annotator.annotate(scene=image, detections=detections, labels=labels)
    return image
