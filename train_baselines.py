import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from Networks.han.utils import *
from Networks.Text_Classification.models.RCNN import RCNN 
from Networks.Text_Classification.models.CNN import CNN 
from Networks.Text_Classification.models.LSTM import LSTMClassifier as LSTM
from Networks.Text_Classification.models.LSTM_Attn import AttentionModel as LSTM_Attn 

# from datasets import HANDataset
from generateDataIters import dataiter
from progiter import ProgIter as tqdm
import itertools
import argparse 
from collections import defaultdict
import sys
import pickle

from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--expname", type=str, help='name of experiment', default='1')
parser.add_argument("--network", type=str, help='name of experiment', default='lstm')

# parser.add_argument("--use_bert_features", action='store_true', help='use bert features')
args = parser.parse_args()

myiter = dataiter(directory=f'sentenced_job_desc_sample_{args.expname}_shuffled')
train_iter, test_iter, TEXT, HASH, ONET_CODE, ONET_CODE_l3, ONET_CODE_l2, ONET_CODE_l1 = myiter.get_iters_title()
print(f'Loaded Data,{len(TEXT.vocab)} \n')

# Model parameters
model_params={
'batch_size':None, 
'output_size':len(ONET_CODE.vocab.freqs.keys()), 
'hidden_size':200,
'vocab_size':len(TEXT.vocab),
'embedding_length':300,
'weights':TEXT.vocab.vectors
}

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 64  # batch size
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
workers = 4  # number of workers for loading data in the DataLoader
epochs = 30  # number of epochs to run without early-stopping
grad_clip = None  # clip gradients at this value
print_freq = 2000  # print training or validation status every __ batches
model_name = args.network#f'{args.expname}_only_l4_3_epochs'
checkpoint = None#f'./Models/han/checkpoint_{model_name}.pth.tar'  # path to model checkpoint, None if none
best_acc = 0.  # assume the accuracy is 0 at first

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

def create_mapping():
    mapping = {1:dict(), 2:dict(), 3:dict()}
    fields = {1:ONET_CODE_l1, 2:ONET_CODE_l2, 3:ONET_CODE_l3}
    for i in range(1,4):
        for onet_code in ONET_CODE.vocab.freqs.keys():
            if(i == 1):
                onet_parent = str(int(int(onet_code)/1000000)*1000000)
            elif(i == 2):
                onet_parent = str(int(int(onet_code)/100000)*100000)
            else:
                onet_parent = str(int(int(onet_code)/1000)*1000)
            
            mapping[i][ONET_CODE.vocab.stoi[onet_code]] = fields[i].vocab.stoi[onet_parent]
    return mapping

mapping = create_mapping()

def main():
    """
    Training and validation.
    """
    global best_acc, epochs_since_improvement, checkpoint, start_epoch

    # DataLoaders
    # train_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'train'), batch_size=batch_size, shuffle=True,
    #                                            num_workers=workers, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'test'), batch_size=batch_size, shuffle=True,
    #                                          num_workers=workers, pin_memory=True)
    
    n_classes = len(ONET_CODE.vocab.freqs)
    # Initialize model or load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        model = nn.DataParallel(model)
        optimizer = checkpoint['optimizer']
        # word_map = checkpoint['word_map']
        # start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        multicls = None
        print(
            '\nLoaded checkpoint from epoch %d, with a previous best accuracy of %.3f.\n' % (start_epoch - 1, best_acc))
    
    else:
        embeddings, emb_size = TEXT.vocab.vectors, 300

        if(args.network == 'lstm'):
            model = LSTM(**model_params)
        elif(args.network == 'lstm_attn'):
            model = LSTM_Attn(**model_params)
        elif(args.network == 'cnn'):
            model = CNN(**model_params)
        else:
            model = RCNN(**model_params)

        multicls = None
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Loss functions
    criterion = nn.CrossEntropyLoss()

#     # Move to device
#     if torch.cuda.device_count() > 1:
#           print("Let's use", torch.cuda.device_count(), "GPUs!")
#           # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#           model = torch.nn.DataParallel(model)
    model = model.cuda()#to(device)
    criterion = criterion.cuda()#to(device)


    # Epochs
    epochs_since_improvement = 0
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        # train_loader = itertools.chain(*train_iters)
        train_loader = train_iter
        train(train_loader=train_loader,
              model=model,
              multicls=multicls,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # One epoch's validation
        val_loader = test_iter
        acc = validate(val_loader=val_loader,
                       model=model,
                       multicls=multicls,
                       criterion=criterion)


        # Did validation accuracy improve?
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        
        # Decay learning rate every epoch
        # adjust_learning_rate(optimizer, 0.5)

        # Save checkpoint
        #if((epoch+1)%2==0):
        save_checkpoint(epoch, model, optimizer, best_acc, epochs_since_improvement, is_best, 'han', model_name)
        # sys.stdout.flush()


def train(train_loader, model, multicls, criterion, optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses_softmax = AverageMeter()
    accs_softmax = AverageMeter()

    start = time.time()

    # Batches
    for i, batch in enumerate(train_loader):

        titles = batch.job_desc
#         print(str(len(sentences_per_document)))
        labels = batch.onet_code
        labels_parent = {3:batch.onet_code_l3, 2:batch.onet_code_l2, 1:batch.onet_code_l1}
        data_time.update(time.time() - start)
        titles = titles.cuda()#to(device)  # (batch_size, sentence_limit, word_limit)
        labels = labels.cuda()#to(device)  # (batch_size)
           
        _, scores = model(titles, titles.size(0))
        # Loss
        loss = criterion(scores, labels)  # scalar
        
        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update
        optimizer.step()

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        losses_softmax.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - start)
        accs_softmax.update(accuracy, labels.size(0))

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\n'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\n'.format(epoch, i, 0,
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, 
                                                                  loss=losses_softmax,
                                                                  acc=accs_softmax))

    print('Loss ({loss.avg:.4f})\n'
            'Accuracy ({acc.avg:.3f})\n'.format(loss=losses_softmax,
                                                acc=accs_softmax))

def validate(val_loader, model, multicls, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: cross entropy loss layer
    :return: validation accuracy score
    """
    model.eval()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses_softmax = AverageMeter()
    accs_softmax = AverageMeter()
    accs_softmax_h = {1:AverageMeter(), 2:AverageMeter(), 3:AverageMeter()}

    start = time.time()

    # Batches
    for i, batch in enumerate(val_loader):

        titles = batch.job_desc
#         print(str(len(sentences_per_document)))
        labels = batch.onet_code
        labels_parent = {3:batch.onet_code_l3, 2:batch.onet_code_l2, 1:batch.onet_code_l1}
        data_time.update(time.time() - start)
        titles = titles.cuda()#to(device)  # (batch_size, sentence_limit, word_limit)
        labels = labels.cuda()#to(device)  # (batch_size)
           
        try:
            _, scores = model(titles, titles.size(0))
        except:
            print(titles.size())
        # Loss
        loss = criterion(scores, labels)  # scalar

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)


        # Keep track of metrics
        losses_softmax.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - start)
        accs_softmax.update(accuracy, labels.size(0))

        predictions = predictions.data.cpu().numpy()
        for level in range(1,4):
            predictions_new = np.vectorize(mapping[level].get)(predictions)
            labels_new = labels_parent[level].numpy()
            accuracy = accuracy_score(predictions_new, labels_new)
            accs_softmax_h[level].update(accuracy, labels.size(0))


        start = time.time()

        if i % print_freq == 0:
            print('Epoch:[{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\n'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\n'.format(i, 0,
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, 
                                                                  loss=losses_softmax,
                                                                  acc=accs_softmax))

    print('Loss ({loss.avg:.4f})\n'
            'Accuracy ({acc.avg:.3f})\n'.format(loss=losses_softmax,
                                                acc=accs_softmax))
    for level in range(1,4):
        print(level, accs_softmax_h[level].avg)

    return accs_softmax.avg


if __name__ == '__main__':
    main()
