import os

class Config:
    device = 'cpu'
    seed = 69
    train_dataset_path = '/kaggle/input/pii-detection-removal-from-educational-data/train.json'
    test_dataset_path = '/kaggle/input/pii-detection-removal-from-educational-data/test.json'
    sample_submission_path = '/home/nischay/PID/Data/sample_submission.csv'
    save_dir = '/tmp/output/1/'
    downsample = 0.45
    truncation = True
    padding = False
    max_length = 3574
    doc_stride = 512
    target_cols = ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM',
                   'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM',
                   'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O']
    load_from_disk = None
    learning_rate = 1e-5
    batch_size = 1
    epochs = 4
    NFOLDS = [0]
    trn_fold = 0
    model_paths = {
        '/kaggle/input/37vp4pjt': 10/10,
        '/kaggle/input/pii-deberta-models/cuerpo-de-piiranha': 2/10,
        '/kaggle/input/pii-deberta-models/cola del piinguuino': 1/10,
        '/kaggle/input/pii-deberta-models/cabeza-del-piinguuino': 5/10,
        '/kaggle/input/pii-deberta-models/cabeza-de-piiranha': 3/10,
        '/kaggle/input/pii-deberta-models/cola-de-piiranha': 1/10,
        '/kaggle/input/pii-models/piidd-org-sakura': 2/10,
        '/kaggle/input/pii-deberta-models/cabeza-de-piiranha-persuade_v0': 1/10,
    }
    converted_path = '/kaggle/input/toonnx2-converted-models'

# Ensure save_dir exists
if not os.path.exists(Config.save_dir):
    os.makedirs(Config.save_dir)


