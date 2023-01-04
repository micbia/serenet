
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import warnings

from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops 
from tensorflow.python.ops import array_ops 
from tensorflow.python.ops import math_ops 

def get_avail_metris(mname):
    if(mname == 'wasserstein_loss'):
        metric = wasserstein_loss
    elif(mname == 'matthews_coef'):
        metric = matthews_coef
    elif(mname == 'precision'):
        metric = precision
    elif(mname == 'recall'):
        metric = recall
    elif(mname == 'r2score'):
        metric = r2score
    elif(mname == 'r2score_loss'):
        metric = r2score_loss
    elif(mname == 'weighted_binary_crossentropy'):
        metric = weighted_binary_crossentropy
    elif(mname == 'balanced_cross_entropy'):
        metric = balanced_cross_entropy
    elif(mname == 'tversky_loss'):
        metric = tversky_loss
    elif(mname == 'iou'):
        metric = iou
    elif(mname == 'iou_loss'):
        metric = iou_loss
    elif(mname == 'dice_coef'):
        metric = dice_coef
    elif(mname == 'dice_coef_loss'):
        metric = dice_coef_loss
    elif(mname == 'phi_coef'):
        metric = phi_coef
    elif(mname == 'focal_loss'):
        metric = focal_loss
    elif(mname == 'mae'):
        metric = tf.keras.metrics.mean_absolute_error
    elif(mname == 'mse'):
        metric = tf.keras.metrics.mean_squared_error
    elif(mname == 'ssiml'):
        metric = ssiml
    elif(mname == 'maeacc'):
        metric = mae_accuracy
    elif(mname == 'local_loss'):
        metric = local_loss
    elif(mname == 'FalseNegativeRate'):
        metric = FalseNegativeRate
    elif(mname == 'FalsePositiveRate'):
        metric = FalsePositiveRate
    elif(mname == 'TruePositiveRate'):
        metric = precision
    elif(mname == 'TNR'):
        metric = specificy
    elif(mname == 'balanced_accuracy'):
        metric = balanced_accuracy
    else:
        ValueError(' No corresponding metric found!')
    return metric

def matthews_coef(y_true, y_pred):
    # thie module is to use on tensors (so for Tensorflow or Keras)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    # define true positive and true negative
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    # define false positive and false negative
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

def confusion_matrix(y_true, y_pred):
    y_true, y_pred = K.clip(y_true, K.epsilon(), 1-K.epsilon()), K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    TP = K.sum(y_pred * y_true)
    FN = K.sum((1 - y_pred) * y_true)
    TN = K.sum((1-y_pred) * (1-y_true))
    FP = K.sum(y_pred * (1-y_true))
    return TN, FP, FN, TP

def precision(y_true, y_pred):
    ''' Precision or a.k.a. True Positive Rate (TPR)
        The fraction of prediciton CORRECTLY guessed as positive (neutral). 
        You want this quantity to be as close as possible to 1 for a perfect prediction.
    '''
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    return TP/(TP + FP + K.epsilon())

def specificy(y_true, y_pred):
    ''' Specificy or a.k.a. True Negative Rate (TNR)
        The fraction of prediciton CORRETLY guessed as negative (ionised). 
        You want this quantity to be as close as possible to 1 for a perfect prediction.
    '''
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    return TN/(FP + TN + K.epsilon())

def FalsePositiveRate(y_true, y_pred):
    ''' False Positive Rate (FPR)
        The fraction of prediciton WRONGLY guessed as positive (neutral). 
        You want this quantity to be as small as possible (almost 0) for a perfect prediction.
    '''
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    return FP/(FP + TN + K.epsilon())

def FalseNegativeRate(y_true, y_pred):
    ''' False Negative Rate (FNR)
        The fraction of prediciton WRONGLY guessed as negative (ionised). 
        You want this quantity to be as small as possible (almost 0) for a perfect prediction.
    '''
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    return FN/(FN + TP + K.epsilon())

def balanced_accuracy(y_true, y_pred):
    TPR = precision(y_true, y_pred)
    TNR = specificy(y_true, y_pred)
    return 0.5*(TPR + TNR) 

def recall(y_true, y_pred):
    # custom recall metrics
    y_true, y_pred = K.clip(y_true, K.epsilon(), 1-K.epsilon()), K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    TP = K.sum(y_pred * y_true)
    FN = K.sum((1 - y_pred) * y_true)
    return TP/(TP + FN + K.epsilon())


def r2score(y_true, y_pred):
    # custom R2-score metrics
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_res/(SS_tot + K.epsilon())

def r2score_loss(y_true, y_pred):
    # a.k.a: fraction of variance unexplained 
    return 1 - r2score(y_true, y_pred)


def weighted_binary_crossentropy(y_true, y_pred):
    # Calculate the binary crossentropy
    one_weight, zero_weight = K.mean(y_true, axis=(0,1)), K.mean(1-y_true, axis=(0,1))
    b_ce = K.binary_crossentropy(y_true, y_pred)
    
    # Apply the weights
    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce
    # Return the mean error
    return K.mean(weighted_b_ce)

def sigmoid_balanced_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, beta=None, name=None):
    nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,labels, logits)
    with ops.name_scope(name, "logistic_loss", [logits, labels]) as name: 
        logits = ops.convert_to_tensor(logits, name="logits") 
        labels = ops.convert_to_tensor(labels, name="labels") 
        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %(logits.get_shape(), labels.get_shape())) 
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype) 
        cond = (logits >= zeros) 
        relu_logits = array_ops.where(cond, logits, zeros) 
        neg_abs_logits = array_ops.where(cond, -logits, logits) 
        #beta=0.5
        balanced_cross_entropy = relu_logits*(1.-beta)-logits*labels*(1.-beta)+math_ops.log1p(math_ops.exp(neg_abs_logits))*((1.-beta)*(1.-labels)+beta*labels)
        #return tf.reduce_mean(balanced_cross_entropy)
        return balanced_cross_entropy

def balanced_cross_entropy(y_true, y_pred):
    """
    To decrease the number of false negatives, set beta~1. To decrease the number of false positives, set beta~0.
    """
    #beta = tf.maximum(tf.reduce_mean(1-y_true), tf.keras.backend.epsilon())
    #beta = tf.maximum(tf.reduce_mean(y_true), tf.keras.backend.epsilon())
    beta = 0.5
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_pred = K.log(y_pred / (1 - y_pred))   # TODO: is this wrong? should it be for y_true? see:https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
    #y_true = K.log(y_true / (1 - y_true))
    return sigmoid_balanced_cross_entropy_with_logits(logits=y_pred, labels=y_true, beta=beta)

def binary_crossentropy(target, output, from_logits=False):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    if from_logits:
        y_pred = tf.clip_by_value(output, K.epsilon(), 1 - K.epsilon())
        return sigmoid_balanced_cross_entropy_with_logits(logits=target, labels=K.log(y_pred/(1-y_pred)), beta=beta)
    
    epsilon_ = tf.constant(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = target * tf.math.log(output + K.epsilon())
    bce += (1 - target) * tf.math.log(1 - output + K.epsilon())
    #return tf.reduce_mean(-bce)
    return -bce

def balanced_binary_crossentropy(target, output, from_logits=False):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    #beta = 1-K.mean(target)
    beta = tf.maximum(K.mean(target), K.epsilon())

    if from_logits:
        y_pred = tf.clip_by_value(output, K.epsilon(), 1 - K.epsilon())
        return sigmoid_balanced_cross_entropy_with_logits(logits=target, labels=K.log(y_pred/(1-y_pred)), beta=beta)
    
    epsilon_ = tf.constant(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = beta * target * tf.math.log(output + K.epsilon())
    bce += (1-beta) * (1 - target) * tf.math.log(1 - output + K.epsilon())
    #return tf.reduce_mean(-bce)
    return -bce

def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.
    Args:
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor. By default, we consider that `output` encodes a probability distribution.
    Returns:
        A tensor.
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    if from_logits:
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
    epsilon_ = tf.constant(K.epsilon(), dtype=output.dtype.base_dtype)

    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = target * tf.math.log(output + K.epsilon())
    bce += (1 - target) * tf.math.log(1 - output + K.epsilon())
    return -bce


def tversky_loss(y_true, y_pred, beta=0.7):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    loss = 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

    return loss

def mae_accuracy(y_true, y_pred):
    # tf.keras.metrics.mean_absolute_percentage_error
    # loss = 100 * abs((y_true - y_pred) / y_true)
    return 1 - K.mean(K.abs((K.flatten(y_true) - K.flatten(y_pred)) / (K.flatten(y_true) + K.epsilon())))


def iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    intersection = K.sum(K.abs(y_true * K.round(K.clip(y_pred, 0, 1))))
    #intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(K.round(K.clip(y_pred, 0, 1))) - intersection
    # avoid divide by zero - if the union is zero, return 1, otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def iou_loss(y_true, y_pred):
    return 1-iou(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def phi_coef(ytrue, ypred): 
    ytrue, ypred = (ytrue.squeeze()).flatten(), (ypred.squeeze()).flatten()

    TP, TN, FP, FN = 0., 0., 0., 0.
    for t, p in zip(ytrue, ypred.round()):
        if(t == 1 and p == 1):
            TP += 1.
        elif(t == 0 and p == 0):
            TN += 1. 
        elif(t == 0 and p == 1):
            FP += 1.
        else:
            FN += 1. 
    
    mcc = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) 
    return mcc


def focal_loss(y_true, y_pred):
    gamma, alpha = 2.0, 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
