from model_torch import *
from sklearn.model_selection import train_test_split
import tools.utils as utils
import tools.sound_tools as sound_tools


def data_load(DATA=os.path.join('data', 'interim')):
    train_data_file = os.path.join(DATA, 'train_data.pkl')
    with open(train_data_file, 'rb') as f:
        train_data = pickle.load(f)
    return train_data

class Data_work():
    def __init__(self, root_dir,
                 n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2):
        """PARAMS
        ======
            training_dir (string) - location where the training data are
            n_mels (integer) - number of Mel buckets to build the spectrograms
            frames (integer) - number of sliding windows to use to slice the Mel spectrogram
        """
        self.root_dir = root_dir
        self.n_mels = n_mels
        self.frames = frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.DATA = os.path.join('data', 'interim')
        self.RAW_DATA = os.path.join('data', 'raw')
        self.PROCESSED_DATA = os.path.join('data', 'processed')

    def build_dataset(self):
        normal_files, abnormal_files = utils.build_files_list(root_dir=os.path.join(self.root_dir), abnormal_dir='NG',
                                                              normal_dir='OK')
        normal_y = np.zeros(len(normal_files))
        # split train data to 80%, 20%
        train_files, test_normal_files, train_labels, test_normal_labels = train_test_split(normal_files, normal_y,
                                                                                            train_size=0.8,
                                                                                            random_state=42,
                                                                                            shuffle=True,
                                                                                            stratify=normal_y)
        abnormal_y = np.ones(len(abnormal_files))
        _, test_abnormal_files, _, test_abnormal_labels = train_test_split(abnormal_files, abnormal_y,
                                                                           train_size=0.6,
                                                                           random_state=42,
                                                                           shuffle=True,
                                                                           stratify=abnormal_y)

        train_files = np.array(train_files)
        train_labels = np.array(train_labels)

        test_files = np.concatenate((test_normal_files, test_abnormal_files), axis=0)
        test_labels = np.concatenate((test_normal_labels, test_abnormal_labels), axis=0)

        dataset = dict({
            'train_files': train_files,
            'test_files': test_files,
            'train_labels': train_labels,
            'test_labels': test_labels
        })
        return dataset

    def save_pkl(self, dataset, name='train_data.pkl'):
        for key, values in dataset.items():
            fname = os.path.join(self.PROCESSED_DATA, key + '.txt')
            with open(fname, 'w') as f:
                for item in values:
                    f.write(str(item))
                    f.write('\n')
        train_data_location = os.path.join(self.DATA, 'train_data.pkl')

        if os.path.exists(train_data_location):
            print('Train data already exists, loading from file...')
            with open(train_data_location, 'rb') as f:
                train_data = pickle.load(f)

        else:
            train_data = sound_tools.generate_dataset(dataset['train_files'], n_mels=self.n_mels, frames=self.frames, n_fft=self.n_fft,
                                                      hop_length=self.hop_length, cnn_type=0)
            print('Saving training data to disk...')
            with open(os.path.join(self.DATA, name), 'wb') as f:
                pickle.dump(train_data, f)
            print('Done.')
        return train_data

    def train_data_get(self):
        dataset = self.build_dataset()
        train_data = self.save_pkl(dataset)

        return train_data


#Data = Data_work("audio_data/train_20200925")
#Data.data_preprocess()
