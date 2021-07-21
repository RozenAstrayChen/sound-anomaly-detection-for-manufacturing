from model_torch import *
import torch
from tqdm import tqdm
import tools.sound_tools as sound_tools
import data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tools import  utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Eval_method():
    def __init__(self, model_path='model/AE.pth', n_mels=64, frames=5, n_fft=1024, hop_length=512, model_type=1):
        self.model_path = model_path
        self.n_mels = n_mels
        self.frames = frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model_type = model_type

        self.prop_cycle = plt.rcParams['axes.prop_cycle']
        self.colors = self.prop_cycle.by_key()['color']
        self.blue = self.colors[1]
        self.red = self.colors[5]
        self.TN = None
        self.FN = None
        self.threshold = 0
    def load_model(self):
        model = autoencoder(self.n_mels * self.frames, self.model_type)
        model.load_state_dict(torch.load(self.model_path))

        self.model = model.to(device)

    def load_eval_data(self, path="audio_data/train_20200925"):
        Data_work = data.Data_work(path)
        dataset = Data_work.build_dataset()

        self.dataset= dataset

    def pred_reconstruction(self):
        self.model.eval()
        # load data
        reconstruction_errors = []
        for index, eval_filename in tqdm(enumerate(self.dataset['test_files']), total=len(self.dataset['test_files'])):
            # Load signal
            signal, sr = sound_tools.load_sound_file(eval_filename)
            eval_features = sound_tools.extract_signal_features(
                signal,
                sr,
                n_mels=self.n_mels,
                frames=self.frames,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            eval_features_device = torch.from_numpy(eval_features).to(device)
            # Get prediction from our autoencoder
            prediction = self.model(eval_features_device).detach().cpu().numpy()

            # Estimate the reconstruction error
            mse = np.mean(np.mean(np.square(eval_features - prediction), axis=1))
            reconstruction_errors.append(mse)

        return reconstruction_errors

    def reconstruction_error_analysis(self, reconstruction_errors):
        y_true = self.dataset['test_labels']

        data = np.column_stack((range(len(reconstruction_errors)), reconstruction_errors))
        bin_width = 0.25
        bins = np.arange(min(reconstruction_errors), max(reconstruction_errors) + bin_width, bin_width)

        fig = plt.figure(figsize=(12, 4))
        plt.hist(data[y_true == 0][:, 1], bins=bins, color=self.blue, alpha=0.6, label='Normal signals', edgecolor='#FFFFFF')
        plt.hist(data[y_true == 1][:, 1], bins=bins, color=self.red, alpha=0.6, label='Abnormal signals',
                 edgecolor='#FFFFFF')
        plt.xlabel("Testing reconstruction error")
        plt.ylabel("# Samples")
        plt.title('Reconstruction error distribution on the testing set', fontsize=16)
        plt.legend()
        #plt.show()

        threshold_min = data[y_true == 0][:, 1].min()
        threshold_max = data[y_true == 0][:, 1].max()

        normal_x, normal_y = data[y_true == 0][:, 0], data[y_true == 0][:, 1]
        abnormal_x, abnormal_y = data[y_true == 1][:, 0], data[y_true == 1][:, 1]
        x = np.concatenate((normal_x, abnormal_x))

        fig, ax = plt.subplots(figsize=(24, 8))
        plt.scatter(normal_x, normal_y, s=15, color='tab:green', alpha=0.3, label='Normal signals')
        plt.scatter(abnormal_x, abnormal_y, s=15, color='tab:red', alpha=0.3, label='Abnormal signals')
        plt.fill_between(x, threshold_min, threshold_max, alpha=0.1, color='tab:orange', label='Threshold range')
        plt.hlines([threshold_min, threshold_max], x.min(), x.max(), linewidth=0.5, alpha=0.8, color='tab:orange')
        plt.legend(loc='upper left')
        plt.title('Threshold range exploration', fontsize=16)
        plt.xlabel('Samples')
        plt.ylabel('Reconstruction error')
        #plt.show()

        return [threshold_min, threshold_max]
    def threshold_judge(self, TN, FN, threshold):
        if self.TN == None and self.FN == None:
            print('init threshold')
            self.TN = TN; self.FN = FN
            self.threshold = threshold

        if self.TN <= TN or self.FN >= FN:
            print('change to new threshold')
            self.threshold = threshold

    def threshold_select(self, t_min, t_max, reconstruction_errors, t_step=0.25):
        '''
        Focus on True/Negative False/Negative
        :param t_min:
        :param t_max:
        :param reconstruction_errors:
        :param t_step:
        :return:
        '''
        thresholds = np.arange(t_min, t_max + t_step, t_step)

        df = pd.DataFrame(columns=['Signal', 'Ground Truth', 'Prediction', 'Reconstruction Error'])
        df['Signal'] = self.dataset['test_files']
        df['Ground Truth'] = self.dataset['test_labels']
        df['Reconstruction Error'] = reconstruction_errors

        FN = []
        FP = []
        for th in thresholds:
            print('thresholds: ', th)
            df.loc[df['Reconstruction Error'] <= th, 'Prediction'] = 0.0
            df.loc[df['Reconstruction Error'] > th, 'Prediction'] = 1.0
            df = utils.generate_error_types(df)

            FN.append(df['FN'].sum())
            FP.append(df['FP'].sum())
            print('TP: %d, TN: %d, FP: %d, FN: %d' % (df['TP'].sum(),
                                                      df['TN'].sum(),
                                                      df['FP'].sum(),
                                                      df['FN'].sum()))
            self.threshold_judge(df['TN'].sum(), df['FN'].sum(), th)
        utils.plot_curves(FP, FN,
                          nb_samples=df.shape[0],
                          threshold_min=t_min,
                          threshold_max=t_max,
                          threshold_step=t_step)
        self.TN = None; self.FN = None;
        print('final threshold is %f' % self.threshold)
        return  self.threshold, df

    def confusion_matrix_result(self, threshold, df):
        df.loc[df['Reconstruction Error'] <= threshold, 'Prediction'] = 0.0
        df.loc[df['Reconstruction Error'] > threshold, 'Prediction'] = 1.0
        df['Prediction'] = df['Prediction'].astype(np.float32)
        df = utils.generate_error_types(df)
        tp = df['TP'].sum()
        tn = df['TN'].sum()
        fn = df['FN'].sum()
        fp = df['FP'].sum()

        from sklearn.metrics import confusion_matrix
        df['Ground Truth'] = 1 - df['Ground Truth']
        df['Prediction'] = 1 - df['Prediction']
        utils.print_confusion_matrix(confusion_matrix(df['Ground Truth'], df['Prediction']),
                                     class_names=['abnormal', 'normal'])


        return {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}
    def metrics_result(self, confusion_matrix):
        precision = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp'])
        recall = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'])
        accuracy = (confusion_matrix['tp'] + confusion_matrix['tn']) / \
                   (confusion_matrix['tp'] + confusion_matrix['tn'] + confusion_matrix['fp'] + confusion_matrix['fn'])
        f1_score = 2 * precision * recall / (precision + recall)

        print(f"""Basic autoencoder metrics:
        - Precision: {precision * 100:.1f}%
        - Recall: {recall * 100:.1f}%
        - Accuracy: {accuracy * 100:.1f}%
        - F1 Score: {f1_score * 100:.1f}%""")


def eval_test():
    Eval = Eval_method()
    Eval.load_model()
    Eval.load_eval_data(path="Fan/")

    reconstruction_errors = Eval.pred_reconstruction()
    threshold_range = Eval.reconstruction_error_analysis(reconstruction_errors)
    select_th, df = Eval.threshold_select(threshold_range[0], threshold_range[1], reconstruction_errors)
    confusion_matrix = Eval.confusion_matrix_result(select_th, df)
    Eval.metrics_result(confusion_matrix)
    plt.show()

eval_test()