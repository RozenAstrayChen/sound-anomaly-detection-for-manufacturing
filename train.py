from model_torch import *
from tools.dataloader import MyDataSet
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train_process():
    def __init__(self, model_dir,
                 n_mels, frame,
                 lr, batch_size, epochs,
                 save_name, model_type):
        """PARAMS
        ======
            training_dir (string) - location where the training data are
            model_dir (string) - location where to store the model artifacts
            n_mels (integer) - number of Mel buckets to build the spectrograms
            frames (integer) - number of sliding windows to use to slice the Mel spectrogram
            lr (float) - learning rate
            batch_size (integer) - batch size
            epochs (integer) - number of epochs
            gpu_count (integer) - number of GPU to distribute the job on
        """
        self.model_dir = model_dir
        self.n_mels = n_mels
        self.frame = frame
        self.LR = lr
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.save_name = save_name
        self.model_type =model_type

    def save_model(self, model, name='model/AE.pth'):
        torch.save(model.state_dict(), name)

    def load_model(self, LR, model_type):
        model = autoencoder(self.n_mels * self.frame, model_type)
        # Model preparation:
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
        # Builds the model:
        return model, criterion, optimizer

    def transform_loader(self, X, y):
        data_set = MyDataSet(X, y)
        data_loader = DataLoader(data_set, batch_size=self.BATCH_SIZE, shuffle=True)

        return data_loader

    def train_model(self, X, y):
        # split to train and valid
        train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.9, shuffle=True)

        train_loader = self.transform_loader(train_X, train_y)
        val_loader = self.transform_loader(val_X, val_y)
        # load model
        self.model, self.criterion, self.optimizer = self.load_model(self.LR, self.model_type)
        self.model = self.model.to(device);

        for epoch in range(0, self.EPOCHS+1):
            sum_loss = 0.0
            # trian part
            self.model.train()
            length = len(train_loader)

            for X, label_y in train_loader:
                X = X.type(torch.FloatTensor).to(device)
                label_y = label_y.type(torch.FloatTensor).to(device)
                pred_y = self.model(X)
                self.optimizer.zero_grad()

                loss = self.criterion(pred_y, label_y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()

            print('---train--- epoch:%d, mean loss: %.05f' % (epoch + 1, sum_loss/length))

            # test part
            with torch.no_grad():
                self.model.eval()
                for X, label_y in val_loader:
                    X = X.type(torch.FloatTensor).to(device)
                    label_y = label_y.type(torch.FloatTensor).to(device)
                    pred_y = self.model(X)
                    loss = self.criterion(pred_y, label_y)
                    print('---test---  Loss: %.05f' % (loss))
                    break

            self.save_model(self.model)

def start_train(data_path="Fan/", model_dir='model/',
                n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2,
                epochs=2000, batch_size=128, lr=0.001,
                save_name='AE', model_type=1
               ):

    Train = Train_process(model_dir,
                          n_mels, frames,
                          lr, batch_size, epochs,
                          save_name, model_type)
    Data = data.Data_work(data_path, n_mels=n_mels, frames=frames, n_fft=n_fft, hop_length=hop_length, power=power)
    train_data = Data.train_data_get()
    Train.train_model(train_data, train_data)

start_train()