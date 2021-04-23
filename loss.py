from keras import backend as K



def residual_error_loss(y_ture, y_pred):
    loss = K.mean(K.sqrt(K.sum(K.square(y_ture-y_pred), axis = 1)) / K.sqrt(K.sum(K.square(y_ture), axis = 1)))
    return loss
