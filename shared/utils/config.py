# config.py

class Config:
    def __init__(self, path, lr, batch_size, shuffle, epochs, optimizer_name, bce_weight, loss_type, val_percent):
        self.path = path
        self.lr = lr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.bce_weight = bce_weight
        self.loss_type = loss_type
        self.val_percent = val_percent

    def to_dict(self):
        return {
            'path': self.path,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'epochs': self.epochs,
            'optimizer': self.optimizer_name,
            'bce_weight': self.bce_weight,
            'loss_type': self.loss_type,
            'val_percent': self.val_percent
        }