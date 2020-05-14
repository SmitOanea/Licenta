from Parameters import Parameters
from FacialDetector import FacialDetector
import pdb
from Visualize import show_detections_with_ground_truth, show_detections_without_ground_truth
import os
import numpy as np


params: Parameters = Parameters()
params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 4   # dimensiunea celulei
params.overlap = 0.3
params.number_positive_examples = 24#6713 * 2   # numarul exemplelor pozitive
params.number_negative_examples = 30#10000  # numarul exemplelor negative
params.threshold = -30000  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = False

params.scaling_ratio = 0.9
params.use_hard_mining = False#True  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite


facial_detector: FacialDetector = FacialDetector(params)

# Pasul 1. Incarcam exemplele pozitive (cropate) si exemple negative generate exemple pozitive
# verificam daca ii avem deja salvati
positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + str(params.dim_hog_cell) +
                                      '_' + str(params.number_positive_examples) + '.npy')
if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features = facial_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)

# exemple negative
negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) +
                                      '_' + str(params.number_negative_examples) + '.npy')
if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    np.save(negative_features_path, negative_features)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

# Pasul 2. Invatam clasificatorul liniar
training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
facial_detector.train_classifier(training_examples, train_labels)


# Pasul 3. (optional) Antrenare cu exemple puternic negative (detectii cu scor >0 din cele 274 de imagini negative)
# Daca implementati acest pas ar trebui sa modificati functia FacialDetector.run()
# astfel incat sa va returneze descriptorii detectiilor cu scor > 0 din cele 274 imagini negative
# completati codul in continuare
# TODO:  (optional)  completeaza codul in continuare

def hard_mining1():
    global params, train_labels, training_examples, facial_detector, \
        positive_features, negative_features
    number_of_iterations = 1
    for iteration in range(1, number_of_iterations + 1):
        hard_negatives_path = os.path.join(params.dir_save_files,
                                           'descriptoriExempleGreuNegative_%d_%d_%d.npy' % (
                                               params.dim_hog_cell,
                                               params.number_hard_negatives,
                                               iteration,
                                           ))
        if os.path.exists(hard_negatives_path):
            print('Load hard negatives iteration %d from file' % iteration)
            hard_negatives = np.load(hard_negatives_path)
        else:
            print('Computing hard negatives iteration %d' % iteration)
            hard_negatives = facial_detector.get_hard_negatives()
            np.save(hard_negatives_path, hard_negatives)
            print('Computed hard negatives iteration %d' % iteration)
        # training_examples = np.concatenate((training_examples, hard_negatives))
        # train_labels = np.concatenate((train_labels, np.zeros(hard_negatives.shape[0])))
        training_examples = np.concatenate((training_examples, hard_negatives))
        train_labels = np.concatenate((train_labels, np.zeros(hard_negatives.shape[0])))
        facial_detector.train_classifier(training_examples, train_labels, iteration=iteration)


def hard_mining2():
    global params, train_labels, training_examples, facial_detector, \
        positive_features, negative_features
    while True:
        pass


if params.use_hard_mining:
    hard_mining1()
    # hard_mining2()


# Pasul 4. Ruleaza detectorul facial pe imaginile de test.
detections, scores, file_names = facial_detector.run()


# Pasul 5. Evalueaza si vizualizeaza detectiile
# Pentru imagini pentru care exista adnotari (cele din setul de date  CMU+MIT)
# folositi functia show_detection_with_ground_truth,
# pentru imagini fara adnotari (cele realizate la curs si laborator)
# folositi functia show_detection_without_ground_truth
if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    print("sadfasdfsadf")
    print("sadfasdfsadf")
    print("sadfasdfsadf")
    print("sadfasdfsadf")
    show_detections_without_ground_truth(detections, scores, file_names, params)
