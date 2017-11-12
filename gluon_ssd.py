import mxnet as mx
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior
from mxnet.contrib.ndarray import MultiBoxTarget
from mxnet.contrib.ndarray import MultiBoxDetection
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon import nn
from mxnet import gluon
import time
from mxnet import autograd as ag
import mxnet.image as image
import os
from PIL import Image, ImageOps
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import sys

class_names = ['pikachu']
num_class = len(class_names)
data_shape = 256


class ToySSD(gluon.Block):
    
    def __init__(self, num_classes, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # 5层，每一层都有着不同sizes，越到后面越小
        self.anchor_sizes = [[0.2, 0.272],
                             [0.37, 0.447],
                             [0.54, 0.619],
                             [0.71, 0.79],
                             [0.88, 0.961]]
        self.anchor_ratios = [[1, 2, 0.5]] * 5
        self.num_classes = num_classes
        with self.name_scope():
            self.body, self.down_samples, self.class_preds, self.box_preds = self.toy_ssd_model(4, num_classes)
    
    @staticmethod
    def class_predictor(num_anchors, num_classes):
        return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)
    
    @staticmethod
    def box_predictor(num_anchors):
        return nn.Conv2D(num_anchors * 4, 3, padding=1)
     
    @staticmethod
    def down_sample(num_filters):
        out = nn.HybridSequential()
        for _ in range(2):
            out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
            out.add(nn.BatchNorm(in_channels=num_filters))
            out.add(nn.Activation('relu'))
        out.add(nn.MaxPool2D(2))
        return out
    
    def body(self):
        out = nn.HybridSequential()
        for f in [16, 32, 64]:
            out.add(self.down_sample(f))
        return out
    
    @staticmethod
    def flatten_prediction(pred):
        return nd.flatten(nd.transpose(pred, axes=(0, 2, 3, 1)))
    
    @staticmethod
    def concat_predictions(preds):
        return nd.concat(*preds, dim=1)

    def toy_ssd_model(self, num_anchors, num_classes):
        """return SSD modules"""
        down_samples = nn.Sequential()
        class_preds = nn.Sequential()
        box_preds = nn.Sequential()

        down_samples.add(self.down_sample(128))
        down_samples.add(self.down_sample(128))
        down_samples.add(self.down_sample(128))

        for scale in range(5):
            class_preds.add(self.class_predictor(num_anchors, num_classes))
            box_preds.add(self.box_predictor(num_anchors))
        return self.body(), down_samples, class_preds, box_preds

    def toy_ssd_forward(self, x, body, downsamples, class_preds, box_preds, sizes, ratios):
        # extracted features
        x = body(x)

        default_anchors = []
        predicted_boxes = []
        predicted_classes = []
        for i in range(5):
            default_anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
            predicted_boxes.append(self.flatten_prediction(box_preds[i](x)))
            predicted_classes.append(self.flatten_prediction(class_preds[i](x)))
            if i < 3:
                x = downsamples[i](x)
            elif i == 3:
                x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))
        return default_anchors, predicted_classes, predicted_boxes
    
    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = self.toy_ssd_forward(x, self.body, self.down_samples,
                                                                                   self.class_preds, self.box_preds,
                                                                                   self.anchor_sizes, self.anchor_ratios
                                                                                   )
        anchors = self.concat_predictions(default_anchors)
        box_preds = self.concat_predictions(predicted_boxes)
        class_preds = self.concat_predictions(predicted_classes)
        class_preds = nd.reshape(class_preds, shape=(0, -1, self.num_classes + 1))
        return anchors, class_preds, box_preds


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pt = F.pick(output, label, axis=self._axis, keepdims=True)
        loss = -self._alpha * ((1 - pt) ** self._gamma) * F.log(pt)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return F.mean(loss, self._batch_axis, exclude=True)


def get_iterators(data_shape, batch_size):
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_train.rec',
        path_imgidx='./data/pikachu_train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_val.rec',
        shuffle=False,
        mean=True)
    return train_iter, val_iter, class_names, num_class


def training_targets(default_anchors, class_predicts, labels):
    class_predicts = nd.transpose(class_predicts, axes=(0, 2, 1))
    z = MultiBoxTarget(*[default_anchors, labels, class_predicts])
    box_target = z[0]  # box offset target for (x, y, width, height)
    box_mask = z[1]  # mask is used to ignore box offsets we don't want to penalize, e.g. negative samples
    cls_target = z[2]  # cls_target is an array of labels for all anchors boxes
    return box_target, box_mask, cls_target


def train():
    batch_size = 32
    train_data, test_data, class_names, num_class = get_iterators(data_shape, batch_size)
    train_data.reshape(label_shape=(3, 5))
    train_data = test_data.sync_label_shape(train_data)

    cls_loss = FocalLoss()
    box_loss = SmoothL1Loss()
    cls_metric = mx.metric.Accuracy()
    box_metric = mx.metric.MAE()

    ctx = mx.gpu()
    try:
        _ = nd.zeros(1, ctx=ctx)
    except mx.base.MXNetError as err:
        print('No GPU enabled, fall back to CPU, sit back and be patient...')
        ctx = mx.cpu()
    net = ToySSD(num_class)

    params = 'ssd_pretrained.params'
    if os.path.exists(params):
        net.load_params(params, ctx=ctx)
    else:
        net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

    start_epoch = 0
    epochs = 150
    log_interval = 10

    for epoch in range(start_epoch, epochs):
        # reset iterator and tick
        train_data.reset()
        cls_metric.reset()
        box_metric.reset()
        tic = time.time()

        try:
            # iterate through all batch
            for i, batch in enumerate(train_data):
                btic = time.time()
                # record gradients
                with ag.record():
                    x = batch.data[0].as_in_context(ctx)
                    y = batch.label[0].as_in_context(ctx)
                    default_anchors, class_predictions, box_predictions = net(x)
                    box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
                    # losses
                    loss1 = cls_loss(class_predictions, cls_target)
                    loss2 = box_loss(box_predictions, box_target, box_mask)
                    # sum all losses
                    loss = loss1 + loss2
                    loss.backward()
                # apply
                trainer.step(batch_size)
                # update metrics
                cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
                box_metric.update([box_target], [box_predictions * box_mask])
                if (i + 1) % log_interval == 0:
                    name1, val1 = cls_metric.get()
                    name2, val2 = box_metric.get()
                    print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f'
                          % (epoch, i, batch_size / (time.time() - btic), name1, val1, name2, val2))
        except KeyboardInterrupt:
            print('# Interrupted, saving params for now.')
            net.save_params('ssd_%d.params' % epoch)
            exit(0)

        # end of epoch logging
        name1, val1 = cls_metric.get()
        name2, val2 = box_metric.get()
        print('[Epoch %d] training: %s=%f, %s=%f' % (epoch, name1, val1, name2, val2))
        print('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))

    # we can save the trained parameters to disk
    net.save_params('ssd_%d.params' % epochs)


def detect_image(img_file):
    if not os.path.exists(img_file):
        print('can not find image: ', img_file)
    img = Image.open(img_file)
    img = ImageOps.fit(img, [data_shape, data_shape], Image.ANTIALIAS)
    print(img)
    origin_img = np.array(img)
    img = origin_img - np.array([123, 117, 104])
    # organize as [batch-channel-height-width]
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :]
    # convert to ndarray
    img = nd.array(img)
    print('input image shape: ', img.shape)

    net = ToySSD(2)
    ctx = mx.cpu()
    # net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    params = 'ssd_pretrained.params'
    net.load_params(params, ctx=ctx)

    anchors, cls_preds, box_preds = net(img.as_in_context(ctx))
    print('anchors', anchors)
    print('class predictions', cls_preds)
    print('box delta predictions', box_preds)

    # convert predictions to probabilities using softmax
    cls_probs = nd.SoftmaxActivation(nd.transpose(cls_preds, (0, 2, 1)), mode='channel')
    # apply shifts to anchors boxes, non-maximum-suppression, etc...
    output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress=True, clip=False)
    print(output)

    mpl.rcParams['figure.figsize'] = (10, 10)
    pens = dict()
    plt.clf()
    plt.imshow(img)

    thresh = 0.45
    for det in output[0]:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        scales = [origin_img.shape[1], origin_img.shape[0]] * 2
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=pens[cid], linewidth=3)
        plt.gca().add_patch(rect)
        text = class_names[cid]
        plt.gca().text(xmin, ymin - 2, '{:s} {:.3f}'.format(text, score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')
    plt.show()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'detect':
        try:
            detect_image(sys.argv[2])
        except Exception as e:
            print(e)
            print('for detect please provide image file path.')
    else:
        print(' `python3 gluon_ssd.py` train for train,'
              '`python3 gluon_ssd.py detect /your/image/path.jpg` for detect.')


