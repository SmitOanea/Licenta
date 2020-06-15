from Parameters import Parameters
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
import os
from skimage.feature import hog
import math


class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None

    def hog(self, img, feature_vector=True):
        return hog(img, orientations=9, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                   cells_per_block=(2, 2), block_norm='L2', transform_sqrt=False, feature_vector=feature_vector)

    def get_positive_descriptors(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpeg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

            # acum o voi redimensiona, pentru ca initial imaginile facute de mine cu telefonul sunt mari
            img = cv.resize(img, (self.params.dim_window, self.params.dim_window))

            hog_img = self.hog(img, feature_vector=True)
            positive_descriptors.append(hog_img)
            if self.params.use_flip_images:
                img = np.flip(img, axis=1)
                hog_img = self.hog(img, feature_vector=True)
                positive_descriptors.append(hog_img)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpeg')
        files = glob.glob(images_path)
        print("images path = ", images_path)
        num_images = len(files)
        print("len files = ", len(files))
        num_negative_per_image = self.params.number_negative_examples // num_images
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            samples = num_negative_per_image
            if i < self.params.number_negative_examples % num_images:
                samples += 1
            for j in range(samples):
                x = np.random.randint(low=0, high=img.shape[0] - self.params.dim_window + 1)
                y = np.random.randint(low=0, high=img.shape[1] - self.params.dim_window + 1)
                curr_img = img[x: x + self.params.dim_window, y: y + self.params.dim_window]
                curr_hog = self.hog(curr_img, feature_vector=True)
                negative_descriptors.append(curr_hog)

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def get_hard_negatives(self):
        images_path = os.path.join(self.params.dir_neg_examples, '*.jpeg')
        files = glob.glob(images_path)

        number_of_hard_examples = self.params.number_hard_negatives
        images_by_path = {}

        def get_img(file: str) -> np.ndarray:
            if file in images_by_path:
                return images_by_path[file]
            images_by_path[file] = cv.imread(file, cv.IMREAD_GRAYSCALE)
            return images_by_path[file]
        hard_negatives = []
        var = 0
        while len(hard_negatives) < 4:#number_of_hard_examples:
            if var%1000==0:
                print("var = ", var)
                print("len(hard_negatives) = ", len(hard_negatives))
            var+=1
            i = np.random.randint(low=0, high=len(files))
            img = get_img(files[i])


            latime_patrat = np.random.randint(low=self.params.dim_window, high=min(3*self.params.dim_window+1, min(img.shape[0], img.shape[1] )) + 1)

            x = np.random.randint(low=0, high=img.shape[0] - latime_patrat + 1)
            y = np.random.randint(low=0, high=img.shape[1] - latime_patrat + 1)
            curr_img = img[x: x + latime_patrat, y: y + latime_patrat]

            curr_img = cv.resize(curr_img, (36, 36))

            curr_hog = self.hog(curr_img, feature_vector=True)
            score = self.best_model.decision_function([curr_hog])[0]
            if score > 0:
                hard_negatives.append(curr_hog)
                if len(hard_negatives) % 100 == 0:
                    print('Computed %d hard negatives' % len(hard_negatives))
        print('Number of hard negatives: %d' % len(hard_negatives))
        return np.array(hard_negatives)


    def train_classifier(self, training_examples, train_labels, iteration=0):
        if iteration == 0:
            negative_examples = self.params.number_negative_examples
        else:
            negative_examples = self.params.number_hard_negatives
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d_%d' %
                                     (self.params.dim_hog_cell, negative_examples,
                                      self.params.number_positive_examples, iteration))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
        # Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()


    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)


        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou


    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        if len(x_out_of_bounds) > 0 or len(y_out_of_bounds):
            print(x_out_of_bounds)
            print(y_out_of_bounds)
            raise RuntimeError("Values out of bounds")
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i]:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j]:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False

        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self, return_descriptors=False):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din
        self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini (imaginile realizate cu voi la curs+laborator).
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera
        o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpeg')
        test_files = glob.glob(test_images_path)
        detections = np.zeros((0, 4), np.int32)  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le optinem
        # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori,
        # pentru fiecare detectie din imagine, numele imaginii va aparea in aceasta lista
        file_names = []
        # w = self.best_model.coef_.T
        # bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)
       
        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d, %s..' % (i, num_test_images, test_files[i]))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)

            '''o redimensionez in caz ca e prea mare, ca sa nu dureze mult testarea'''
            '''if im[0]>420:
                raport = img.shape[0] / 420.0
                new_height = int(img.shape[0] / raport)
                new_width = int(img.shape[1] / raport)
                print("\n\nredimensionez imaginea de test ASTFEL INCAT SA O PASTREZ")
                print("new_hight = ", new_height)
                print("new_width = ", new_width)

                img_de_pastrat = cv.resize(img, (new_width, new_height))
                print("test_files[i] = ", test_files[i])
                cv.imwrite("micsorata_" + test_files[i] + ".jpeg", img_de_pastrat)'''
            if img.shape[0] < img.shape[1]:
                if img.shape[0] > self.params.width_redimensionare:

                    raport = img.shape[0] / (1.0*self.params.width_redimensionare)
                    new_height = int(img.shape[0] / raport)
                    new_width = int(img.shape[1] / raport)
                    print("\n\nredimensionez imaginea de test pentru ca e foarte mare. Noile dimensiuni sunt: ")
                    print("new_hight = ", new_height)
                    print("new_width = ", new_width)
                    print("vechile dimensiuni erau ", img.shape, "\n\n")

                    img = cv.resize(img, (new_width, new_height))
            else:
                print("intru pe ELSE")
                if img.shape[1] > self.params.width_redimensionare:
                    raport = img.shape[1] / (1.0 * self.params.width_redimensionare)
                    new_height = int(img.shape[0] / raport)
                    new_width = int(img.shape[1] / raport)
                    print("\n\nredimensionez imaginea de test pentru ca e foarte mare. Noile dimensiuni sunt: ")
                    print("new_hight = ", new_height)
                    print("new_width = ", new_width)
                    print("vechile dimensiuni erau ", img.shape, "\n\n")

                    img = cv.resize(img, (new_width, new_height))


            image_resizing_multiplier = self.params.scaling_ratio

            resize_multiplier = 1
            current_detections = []
            current_scores = []
            h, w = img.shape[0], img.shape[1]
            nr_detectii_gasite = 0
            pas = 0
            while nr_detectii_gasite<3:
                pas+=1
                h = math.floor(img.shape[0] * resize_multiplier)
                w = math.floor(img.shape[1] * resize_multiplier)
                if h < 30 or w < 30:
                    break
                resized_img = cv.resize(img, (w, h))
                hog_cell = self.params.dim_hog_cell
                img_hog = self.hog(resized_img, feature_vector=False)
                nh, nw = img_hog.shape[0], img_hog.shape[1]
                win_h = self.params.dim_window // hog_cell - 1
                win_w = win_h
                limita_stanga = -1
                limita_sus = -1
                pas_i = 0
                for ih in range(0, nh - win_h + 1):
                    if limita_sus != -1:
                        ih = limita_sus
                        ih+=pas_i
                        pas_i+=1

                    for iw in range(0, nw - win_w + 1):
                        if limita_sus != -1:
                            ih = limita_stanga
                            ih += pas_i
                            pas_i += 1

                        window = img_hog[ih: ih + win_h, iw: iw + win_w, :]
                        window = np.ravel(window)

                        score = self.best_model.decision_function([window])[0]
                        if score > self.params.threshold:#am gasit o detectie
                            real_ih = math.floor(ih * hog_cell / resize_multiplier)
                            real_iw = math.floor(iw * hog_cell / resize_multiplier)
                            real_ihp = math.floor((ih + win_h) * hog_cell / resize_multiplier)
                            real_iwp = math.floor((iw + win_w) * hog_cell / resize_multiplier)
                            current_detections.append([real_iw, real_ih, real_iwp, real_ihp])
                            current_scores.append(score)
                            nr_detectii_gasite+=1
                            #break#experimental. De scos daca merge prost
                            if nr_detectii_gasite>2:
                                break

                resize_multiplier *= image_resizing_multiplier

            current_detections = np.reshape(np.array(current_detections, np.int32), (-1, 4))
            current_scores = np.reshape(np.array(current_scores, np.float64), -1)
            current_detections, current_scores = self.non_maximal_suppression(current_detections,
                                                                              current_scores,
                                                                              img.shape[0:2])
            print('Found %d detections' % len(current_detections))
            detections = np.concatenate((detections, current_detections))
            scores = np.concatenate((scores, current_scores))
            file_names += [ntpath.basename(test_files[i])] * len(current_detections)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        print("\n\n")
        return detections, scores, np.array(file_names)

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision


    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
