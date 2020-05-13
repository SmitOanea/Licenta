from Parameters import Parameters
import pickle

params: Parameters = Parameters()
params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 6   # dimensiunea celulei
params.overlap = 0.3
params.number_positive_examples = 6713   # numarul exemplelor pozitive
params.number_negative_examples = 10000  # numarul exemplelor negative
params.threshold = 0  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.scaling_ratio = 0.9
params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = False  # adauga imaginile cu fete oglindite

pickle.dump(params, open('../parametri/params_1', 'wb'))
