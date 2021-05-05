import torch
from torch.utils.data import Dataset, DataLoader, sampler
import pandas as pd
from functools import lru_cache
import numpy as np
import logging


class DataSplit:
    r"""
    Auxiliary class for splitting data in train, test and validation.
    """

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False):
        r"""
        Initializes the DataSplit.

        Params
        =======
            dataset (Dataset): Data set to split.
            test_train_split (float): Share of training data from the data set.
            val_train_split (float): Share of validation data of the training data.
            shuffle (bool): Boolean indicating whether to shuffle the training and test data.

        Attributes
        ===========
            indices (List): List of all indices in the data set.
            train_indices (List): List of indices in the training data set.
            train_sampler (sampler): Sampler for the training data set.
            val_sampler (sampler): Sampler for the validation data set.
            test_sampler (sampler): Sampler for the test data set.

        """
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = train_indices[: validation_split], train_indices[validation_split:]

        self.train_sampler = sampler.SubsetRandomSampler(self.train_indices)
        self.val_sampler = sampler.SubsetRandomSampler(self.val_indices)
        self.test_sampler = sampler.SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, pin_memory=False,
                                       sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = DataLoader(self.dataset, batch_size=batch_size, pin_memory=False,
                                     sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = DataLoader(self.dataset, batch_size=batch_size, pin_memory=False,
                                      sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader


class InsertionData(Dataset):
    r"""
    Data set to learn to approximate insertion decisions.

    """

    def __init__(self, csv_path, feature_list=None, label_list=None):
        """
        Initializes the data set.

        Params
        =======
            csv_path (string): Path to csv file.
            feature_list (List): List of features to consider.
            label_list (List): List of labels to predict.

        Attributes
        ===========
            data (pandas.DataFrame): Data frame containing the training data.
            samples (array): Unscaled samples that serves as input to the dnn once scaled.
            labels (array): Lables that the dnn tries to predict.
            normalized_samples (array): Scaled samples that serve as the input to the dnn.

        """
        # Read the csv file
        self.data = pd.read_csv(csv_path, header=0, sep=';')

        # Set up samples
        if feature_list is not None:
            self.samples = np.asarray(self.data[feature_list])
        else:
            self.samples = np.asarray(self.data.loc[:, (self.data.columns != 'insertion_index_i') &
                                                       (self.data.columns != 'insertion_index_j') &
                                                       (self.data.columns != 'insertion_cost')])
        # Set up labels labels
        if label_list is not None:
            self.labels = np.asarray(self.data[label_list])
        else:
            self.labels = np.asarray(self.data[["insertion_index_i", "insertion_index_j"]])
        # Normalize
        self.normalized_samples = scale(self.samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get sample from the pandas df
        sample = self.normalized_samples[index]
        # Get label
        label = self.labels[index]
        return sample, label


def scale(X, x_min=-1, x_max=1):
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom / denom


class EarlyStopping:
    r"""
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, file_name="checkpoint"):
        """
        Initializes the early stopping criterion.

        Params
        =======
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            file_name (str): File name of the checkpoint output file.

        Attributes
        ===========
            counter (int): Counter of epochs without improvement
            best_score (float): Best score achieved so far.
            early_stop (bool): Indicates whether the early stopping criterion is fulfilled.
            val_loss_min (float): Minimal validation loss observed so far.

        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.file_name = file_name

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        r"""
        Saves model when validation loss decreases.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.file_name + '.pt')
        self.val_loss_min = val_loss


class DeepInsertionModel(torch.nn.Module):
    r"""
    Deep neural net to predict insertion points for restaurant and customer.
    """

    def __init__(self, encoder_shape, fc1_shapes, fc2_shapes):
        r"""
        Initializes the network.

        Params
        =======
            encoder_shape (int): Size of the encorder layer.
            fc1_shapes (List): List of layer sizes for the first fully connected head.
            fc2_shapes (List): List of layer sizes for the second fully connected head.

        """

        super(DeepInsertionModel, self).__init__()
        self.encoder1 = torch.nn.Sequential(torch.nn.Linear(encoder_shape, encoder_shape),
                                            torch.nn.BatchNorm1d(encoder_shape),
                                            torch.nn.ReLU())
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(encoder_shape, fc1_shapes[0]),
                                       torch.nn.BatchNorm1d(fc1_shapes[0]),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(fc1_shapes[0], fc1_shapes[1]),
                                       torch.nn.BatchNorm1d(fc1_shapes[1]),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(fc1_shapes[1], fc1_shapes[2]),
                                       torch.nn.BatchNorm1d(fc1_shapes[2]),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(fc1_shapes[2], fc1_shapes[3]),
                                       torch.nn.BatchNorm1d(fc1_shapes[3]),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(fc1_shapes[3], fc1_shapes[4]))
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(encoder_shape + fc1_shapes[4], fc2_shapes[0]),
                                       torch.nn.BatchNorm1d(fc2_shapes[0]),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(fc2_shapes[0], fc2_shapes[1]),
                                       torch.nn.BatchNorm1d(fc2_shapes[1]),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(fc2_shapes[1], fc2_shapes[2]),
                                       torch.nn.BatchNorm1d(fc2_shapes[2]),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(fc2_shapes[2], fc2_shapes[3]),
                                       torch.nn.BatchNorm1d(fc2_shapes[3]),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(fc2_shapes[3], fc2_shapes[4]))
        self.out = torch.nn.Softmax()

    def forward(self, x):
        r"""
        Forwards the input to the output likelihood for both insertion points.
        """
        v = self.encoder1.forward(x)
        v1 = self.fc1(v)
        v1 = self.out(v1)
        v2 = self.fc2(torch.cat([v, v1], 1))
        v2 = self.out(v2)
        return v1, v2


if __name__ == '__main__':
    """ Train the dnn """
    import simplejson as json

    torch.manual_seed(0)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initialize model")
    model = DeepInsertionModel(encoder_shape=65, fc1_shapes=[128, 128, 128, 128, 11],
                               fc2_shapes=[128, 128, 128, 128, 12])

    model.to(device)

    print("Load data.")
    dataset = InsertionData(csv_path="training_data/iowa_training_data.csv")
    split = DataSplit(dataset, shuffle=True, val_train_split=0.0)
    trainloader, _, testloader = split.get_split(batch_size=200, num_workers=4)

    print("Start training.")
    # set up optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=320, T_mult=1, eta_min=0,
                                                                     last_epoch=-1)

    # early stopping
    early_stopping = EarlyStopping(patience=400, verbose=True, file_name="deep_insertion_checkpoint")

    # start training
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    lr = []
    epochs = 320
    iters = len(trainloader)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss_i = 0.0
        running_loss_j = 0.0
        running_accuracy_i = 0.0
        running_accuracy_j = 0.0
        for index, (inputs, labels) in enumerate(trainloader):
            # forward input and then backpropagate
            inputs = inputs.float()  # vanilla
            labels_i, labels_j = torch.split(labels.long(), 1, dim=1)
            labels_i = labels_i.flatten()
            labels_j = labels_j.flatten()
            opt.zero_grad()
            preds_i, preds_j = model.forward(inputs)
            loss_i = criterion(preds_i, labels_i)
            loss_j = criterion(preds_j, labels_j)
            loss = (1 / 3) * loss_i + (2 / 3) * loss_j
            loss.backward()
            running_loss_i += loss_i.item()
            running_loss_j += loss_j.item()
            opt.step()
            scheduler.step(epoch + index / iters)
            lr.append(opt.param_groups[0]["lr"])
            # calculate accuracies
            scores_i, predictions_i = torch.max(preds_i.data, 1)
            train_correct_i = int(sum(predictions_i == labels_i))  # labels.size(0) returns int
            acc_i = round((train_correct_i / predictions_i.size()[0]), 2)
            running_accuracy_i += acc_i
            scores_j, predictions_j = torch.max(preds_j.data, 1)
            train_correct_j = int(sum(predictions_j == labels_j))  # labels.size(0) returns int
            acc_j = round((train_correct_j / predictions_j.size()[0]), 2)
            running_accuracy_j += acc_j

        test_loss_i = 0
        test_loss_j = 0
        test_accuracy_i = 0
        test_accuracy_j = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                # inputs = inputs.float().unsqueeze(2).permute(1, 0, 2)  #rwa
                inputs = inputs.float()  # vanilla
                labels_i, labels_j = torch.split(labels.long(), 1, dim=1)
                labels_i = labels_i.flatten()
                labels_j = labels_j.flatten()
                preds_i, preds_j = model.forward(inputs)
                batch_loss_i = criterion(preds_i, labels_i)
                batch_loss_j = criterion(preds_j, labels_j)
                test_loss_i += batch_loss_i.item()
                test_loss_j += batch_loss_j.item()

                # calculate accuracies
                scores_i, predictions_i = torch.max(preds_i.data, 1)
                train_correct_i = int(sum(predictions_i == labels_i))  # labels.size(0) returns int
                acc_i = round((train_correct_i / predictions_i.size()[0]), 2)
                test_accuracy_i += acc_i
                scores_j, predictions_j = torch.max(preds_j.data, 1)
                train_correct_j = int(sum(predictions_j == labels_j))  # labels.size(0) returns int
                acc_j = round((train_correct_j / predictions_j.size()[0]), 2)
                test_accuracy_j += acc_j

        # write out loss and accuracy
        train_losses.append([running_loss_i / len(trainloader), running_loss_j / len(trainloader)])
        train_accuracies.append([running_accuracy_i / len(trainloader), running_accuracy_j / len(trainloader)])
        test_losses.append([test_loss_i / len(testloader), test_loss_j / len(testloader)])
        test_accuracies.append([test_accuracy_i / len(testloader), test_accuracy_j / len(testloader)])
        print(f"Epoch {epoch}/{epochs}.. "
              f"Train loss i : {running_loss_i / len(trainloader):.3f}.. "
              f"Train loss j : {running_loss_j / len(trainloader):.3f}.. "
              f"Train acc i: {running_accuracy_i / len(trainloader):.3f}.. "
              f"Train acc j: {running_accuracy_j / len(trainloader):.3f}.. "
              f"Test loss i: {test_loss_i / len(testloader):.3f}.. "
              f"Test loss j: {test_loss_j / len(testloader):.3f}.. "
              f"Test acc i: {test_accuracy_i / len(testloader):.3f}.. "
              f"Test acc j: {test_accuracy_j / len(testloader):.3f}.. ")
        with open("training_results/deep_insertion_train_loss_iowa", 'w') as f:
            json.dump(train_losses, f)
        with open("training_results/deep_insertion_train_acc_iowa", 'w') as f:
            json.dump(train_accuracies, f)
        with open("training_results/deep_insertion_test_loss_iowa", 'w') as f:
            json.dump(test_losses, f)
        with open("training_results/deep_insertion_test_acc_iowa", 'w') as f:
            json.dump(test_accuracies, f)
        with open("training_results/deep_insertion_lr_iowa", 'w') as f:
            json.dump(lr, f)

        early_stopping((test_loss_i + test_loss_j) / 2 / len(testloader), model)
        if early_stopping.early_stop:
            break

    print('Finished Training')
