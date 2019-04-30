import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc


class Metrics(Callback):

    # TODO: refactor the code
    # https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras

    def __init__(self, training_data):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.X_train = training_data[0]
        self.Y_train = training_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        y_pred = self.model.predict(self.X_train)

        roc = roc_auc_score(self.Y_train, y_pred, average='micro')
        roc_val = roc_auc_score(val_predict, val_targ, average='micro')

        print('\t   - val_f1: {} - val_precision: {} - val_recall {}'
              .format(
                        str(round(_val_f1, 4)),
                        str(round(_val_precision, 4)),
                        str(round(_val_recall, 4))
                ))
        print('\t   - roc-auc: {} - roc-auc_val: {}'
              .format(
                        str(round(roc, 4)),
                        str(round(roc_val, 4))
                ),
                end=100 * ' ' + '\n')

        return


def calculate_accuracy_score(model, X_test, Y_test, batch_size):
    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
    print('score: %.2f' % score)
    print('acc: %.2f' % acc)

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

    for x in range(len(X_test)):
        result = model.predict(X_test[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

        if np.argmax(result) == np.argmax(Y_test[x]):
            if np.argmax(Y_test[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.argmax(Y_test[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print('pos_acc', pos_correct/pos_cnt*100, '%')
    print('neg_acc', neg_correct/neg_cnt*100, '%')

    print('\n')
