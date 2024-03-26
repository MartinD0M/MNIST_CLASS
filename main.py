from dataset import DATAWRAPPER
from classification import ClassifierCNN, ClassifierLinear
from utils import show_batch
import torch

datawrapper = DATAWRAPPER(download = True)
model = ClassifierCNN()
TrainLoader = datawrapper.get_LOADER_TRAIN(batch_size = 32)
ValLoader = datawrapper.get_LOADER_TEST()


def mainShow():
    iterator = iter(TrainLoader)
    imgs, _ = next(iterator)
    show_batch(imgs, model.to_label(model.forward(x = imgs)))


def mainLoad(path_to_file : str = 'saved_models/saved_CNN_99.pth'):
    check = torch.load(path_to_file)
    model.load_state_dict(check['model_state'])


def mainTrain():
    model.train(TrainLoader = TrainLoader)


if __name__ == '__main__':
    #mainTrain()
    mainLoad()
    mainShow()
