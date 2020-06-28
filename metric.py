from sklearn.metrics import roc_auc_score, roc_curve

class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = torch.flatten(torch.sigmoid(y_pred)).data.cpu().numpy()
        self.y_true = np.append(self.y_true, y_true)
        self.y_pred = np.append(self.y_pred, y_pred)
        self.score = roc_auc_score(self.y_true, self.y_pred)

    @property
    def avg(self):
        return self.score