import cv2 as cv
import os
import numpy as np
import pdb
import ntpath
import glob
from Parameters import Parameters


def show_detections_without_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """
    print('No annotations')
    test_images_path = os.path.join(params.dir_test_examples, '*.jpeg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:#citesc imaginea in dimensiunea ei originala(ne redimensionata)
        imagine_originala = cv.imread(test_file)
        if imagine_originala.shape[0] < imagine_originala.shape[1]:
            if imagine_originala.shape[0] > params.width_redimensionare:
                raport = imagine_originala.shape[0] / (1.0*params.width_redimensionare)
                new_height = int(imagine_originala.shape[0] / raport)
                new_width = int(imagine_originala.shape[1] / raport)
                image = cv.resize(imagine_originala, (new_width, new_height))
        else:
            if imagine_originala.shape[1] > params.width_redimensionare:
                raport = imagine_originala.shape[1] / (1.0*params.width_redimensionare)
                new_height = int(imagine_originala.shape[0] / raport)
                new_width = int(imagine_originala.shape[1] / raport)
                image = cv.resize(imagine_originala, (new_width, new_height))


        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            #(detection[0], detection[1]), (detection[2], detection[3])




            detected_image = image[detection[1]:detection[3], detection[0]:detection[2]]
            x1 = int(detection[1]*raport)
            x2 = int(detection[3]*raport)
            y1 = int(detection[0]*raport)
            y2 = int(detection[2]*raport)
            original_detected_image = imagine_originala[detection[1]:detection[3], detection[0]:detection[2]]


            cv.imwrite( params.dir_extracted_detections +  "/pentruExtrasCulori" + short_file_name + ".jpeg", detected_image)


            #cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.rectangle(imagine_originala, (y1, x1), (y2, x2), (0,0,255), thickness=1)
            #cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
            #           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        #cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        cv.imwrite(os.path.join(params.dir_save_files, "detections__originala_" + short_file_name), imagine_originala)

        if params.select_ROI:

            #cv.imshow('image', np.uint8(image))
            # Select ROI
            r = cv.selectROI(image)

            # Crop image
            imCrop = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            #redimensionez decuparea ROI, pentru ca probabil nu am decupat-o exact patrata
            imCrop = cv.resize(imCrop, (params.dim_window, params.dim_window))
            cv.imwrite(os.path.join(params.dir_ROI_images, "ROI_" + short_file_name), imCrop)

            #cv.imshow("y", imCrop)
            #cv.waitKey(0)
        else:
            print('Apasa orice tasta pentru a continua...')
            cv.imshow('image', np.uint8(image))
            #cv.imshow('image', np.uint8(imagine_originala))
            cv.waitKey(0)

'''
Nu o folosesc momenta, si o comentez ca sa nu o mai confund cu functia de sus

def show_detections_with_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate. Deseneaza bounding box-urile prezice si cele corecte.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """
    print('With annotations')
    ground_truth_bboxes = np.loadtxt(params.path_annotations, dtype='str')
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]

        # show ground truth bboxes
        for detection in annotations:
            cv.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])),
                         (0, 255, 0), thickness=1)

        cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        
        print('Apasa orice tasta pentru a continua...')
        cv.waitKey(0)

'''
