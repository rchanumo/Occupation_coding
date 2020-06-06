import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from Networks.han.model import HierarchialAttentionNetwork
from Networks.han.utils import *
# from datasets import HANDataset
from generateDataIters import dataiter
from progiter import ProgIter as tqdm
import itertools
import argparse 
from collections import defaultdict
import sys
import pickle
from Networks.multiclassifier import multiclassifier

parser = argparse.ArgumentParser()
parser.add_argument("--expname", type=str, help='name of experiment', default='1')
parser.add_argument("--attn", type=int, help='use attention', default=1)
parser.add_argument("--use_hierarchy", action='store_true', help='use hierarchy')

args = parser.parse_args()

myiter = dataiter(directory=f'sentenced_job_desc_sample_{args.expname}_shuffled')
train_iter, test_iter, unseen_iter, onet_iter, TEXT, HASH, ONET_CODE, ONET_CODE_l3, ONET_CODE_l2, ONET_CODE_l1 = myiter.get_iters_softmax()
print('Loaded Data\n')


# Model parameters
word_rnn_size = 100  # word RNN size
sentence_rnn_size = 100  # character RNN size
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
dropout = 0.3  # dropout
attn = args.attn
fine_tune_word_embeddings = False  # fine-tune word embeddings?
use_hierarchy = args.use_hierarchy

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 64  # batch size
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
workers = 4  # number of workers for loading data in the DataLoader
epochs = 30  # number of epochs to run without early-stopping
grad_clip = None  # clip gradients at this value
print_freq = 2000  # print training or validation status every __ batches
model_name = f'{args.expname}_only_l4'
checkpoint = None#f'./Models/han/checkpoint_{model_name}.pth.tar'  # path to model checkpoint, None if none
best_acc = 0.  # assume the accuracy is 0 at first

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

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
        if(args.use_hierarchy):
            multicls = torch.load(f'./Models/han/multicls_{model_name}.pt')
        print(
            '\nLoaded checkpoint from epoch %d, with a previous best accuracy of %.3f.\n' % (start_epoch - 1, best_acc))
    
    else:
        embeddings, emb_size = TEXT.vocab.vectors, 300

        model = HierarchialAttentionNetwork(n_classes=n_classes,
                                            vocab_size=len(TEXT.vocab),
                                            emb_size=emb_size,
                                            word_rnn_size=word_rnn_size,
                                            sentence_rnn_size=sentence_rnn_size,
                                            word_rnn_layers=word_rnn_layers,
                                            sentence_rnn_layers=sentence_rnn_layers,
                                            word_att_size=word_att_size,
                                            sentence_att_size=sentence_att_size,
                                            dropout=dropout,
                                            attn=attn)
        model.sentence_attention.word_attention.init_embeddings(
            embeddings)  # initialize embedding layer with pre-trained embeddings
        model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune_word_embeddings)  # fine-tune
        multicls = None
        if(args.use_hierarchy):
            multicls = multiclassifier(2*sentence_rnn_size, [len(ONET_CODE_l1.vocab.freqs), len(ONET_CODE_l2.vocab.freqs), len(ONET_CODE_l3.vocab.freqs)], dropout)
            multicls.cuda()
            optimizer = optim.Adam(params=list(filter(lambda p: p.requires_grad, model.parameters()))+list(multicls.parameters()), lr=lr)
        else:
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

        if((epoch+1)%15 == 0):
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
            if(args.use_hierarchy):
                torch.save(multicls, f'./Models/han/multicls_{model_name}.pt')
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
    losses_softmax = []
    for _ in range(myiter.T_depth):
        losses_softmax.append(AverageMeter())
    accs_softmax = []
    for _ in range(myiter.T_depth):
        accs_softmax.append(AverageMeter())

    start = time.time()

    omega = {1:1, 2:1, 3:1, 4:1}
    # Batches
    for i, batch in enumerate(train_loader):

        documents, sentences_per_document, words_per_sentence = batch.job_desc
#         print(str(len(sentences_per_document)))
        labels = batch.onet_code
        labels_parent = {3:batch.onet_code_l3, 2:batch.onet_code_l2, 1:batch.onet_code_l1}
        data_time.update(time.time() - start)
        documents = documents.cuda()#to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.cuda()#to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.cuda()#to(device)  # (batch_size, sentence_limit)
        labels = labels.cuda()#to(device)  # (batch_size)
       
        for level in range(3, 0, -1):
            labels_parent[level] = labels_parent[level].cuda()            

        # Forward prop.
        doc_features, scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                     words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)
        
        # Loss
        loss = criterion(scores, labels)  # scalar
        # loss = None
        loss_l4 = loss.item()

        if(args.use_hierarchy):
            scores_parent = multicls(doc_features)
            # loss *= omega[4]
            for level in range(3, 0, -1):
                # print('Scores Size: ', scores.size())
                loss_temp = omega[level]*criterion(scores_parent[level-1], labels_parent[level])
                if(loss is None):
                    loss = loss_temp
                else:
                    loss += loss_temp
                _, predictions = scores_parent[level-1].max(dim=1)  # (n_documents)
                correct_predictions = torch.eq(predictions, labels_parent[level]).sum().item()
                accuracy = correct_predictions / labels_parent[level].size(0)
                # Keep track of metrics
                losses_softmax[level-1].update(loss_temp.item(), labels_parent[level].size(0))
                batch_time.update(time.time() - start)
                accs_softmax[level-1].update(accuracy, labels_parent[level].size(0))


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
        losses_softmax[3].update(loss_l4, labels.size(0))
        batch_time.update(time.time() - start)
        accs_softmax[3].update(accuracy, labels.size(0))

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\n'
                  'Loss l1 {loss_l1.val:.4f} ({loss_l1.avg:.4f})\n'
                  'Loss l2 {loss_l2.val:.4f} ({loss_l2.avg:.4f})\n'
                  'Loss l3 {loss_l3.val:.4f} ({loss_l3.avg:.4f})\n'
                  'Loss l4 {loss_l4.val:.4f} ({loss_l4.avg:.4f})\n'
                  'Accuracy l1 {acc_l1.val:.3f} ({acc_l1.avg:.3f})\n'
                  'Accuracy l2 {acc_l2.val:.3f} ({acc_l2.avg:.3f})\n'
                  'Accuracy l3 {acc_l3.val:.3f} ({acc_l3.avg:.3f})\n'
                  'Accuracy l4 {acc_l4.val:.3f} ({acc_l4.avg:.3f})\n'.format(epoch, i, 0,
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, 
                                                                  loss_l1=losses_softmax[0],
                                                                  loss_l2=losses_softmax[1],
                                                                  loss_l3=losses_softmax[2],
                                                                  loss_l4=losses_softmax[3],
                                                                  acc_l1=accs_softmax[0],
                                                                  acc_l2 = accs_softmax[1],
                                                                  acc_l3 = accs_softmax[2],
                                                                  acc_l4 = accs_softmax[3]))

    print('Loss l1 ({loss_l1.avg:.4f})\n'
                  'Loss l2 ({loss_l2.avg:.4f})\n'
                  'Loss l3 ({loss_l3.avg:.4f})\n'
                  'Loss l4 ({loss_l4.avg:.4f})\n'
                  'Accuracy l1 ({acc_l1.avg:.3f})\n'
                  'Accuracy l2 ({acc_l2.avg:.3f})\n'
                  'Accuracy l3 ({acc_l3.avg:.3f})\n'
                  'Accuracy l4 ({acc_l4.avg:.3f})\n'.format(loss_l1=losses_softmax[0],
                                                                  loss_l2=losses_softmax[1],
                                                                  loss_l3=losses_softmax[2],
                                                                  loss_l4=losses_softmax[3],
                                                                  acc_l1=accs_softmax[0],
                                                                  acc_l2 = accs_softmax[1],
                                                                  acc_l3 = accs_softmax[2],
                                                                  acc_l4 = accs_softmax[3]))


def validate(val_loader, model, multicls, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: cross entropy loss layer
    :return: validation accuracy score
    """
    model.eval()

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    # losses = AverageMeter()  # cross entropy loss
    # accs = AverageMeter()  # accuracies
    losses_softmax = []
    for _ in range(myiter.T_depth):
        losses_softmax.append(AverageMeter())
    accs_softmax = []
    for _ in range(myiter.T_depth):
        accs_softmax.append(AverageMeter())
    start = time.time()

    # Batches
    for i, batch in enumerate(val_loader):

        documents, sentences_per_document, words_per_sentence = batch.job_desc
        labels = batch.onet_code
        labels_parent = {3:batch.onet_code_l3, 2:batch.onet_code_l2, 1:batch.onet_code_l1}
        data_time.update(time.time() - start)
        documents = documents.cuda()#to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.cuda()#to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.cuda()#to(device)  # (batch_size, sentence_limit)
        labels = labels.cuda()#to(device)  # (batch_size)

        for level in range(3, 0, -1):
            labels_parent[level] = labels_parent[level].cuda()  

        # Forward prop.
        doc_features, scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                     words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

        # Loss
        loss = criterion(scores, labels)

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        losses_softmax[3].update(loss.item(), labels.size(0))
        batch_time.update(time.time() - start)
        accs_softmax[3].update(accuracy, labels.size(0))

        if(args.use_hierarchy):
            scores_parent = multicls(doc_features)
            for level in range(3, 0, -1):
                # print('Scores Size: ', scores.size())
                loss = criterion(scores_parent[level-1], labels_parent[level])
                _, predictions = scores_parent[level-1].max(dim=1)  # (n_documents)
                correct_predictions = torch.eq(predictions, labels_parent[level]).sum().item()
                accuracy = correct_predictions / labels_parent[level].size(0)
                # Keep track of metrics
                losses_softmax[level-1].update(loss.item(), labels_parent[level].size(0))
                batch_time.update(time.time() - start)
                accs_softmax[level-1].update(accuracy, labels_parent[level].size(0))

        start = time.time()

        # Print training status
        if i % print_freq == 0:
            print('batch: [{0}]\n'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\n'
                  'Loss l1 {loss_l1.val:.4f} ({loss_l1.avg:.4f})\n'
                  'Loss l2 {loss_l2.val:.4f} ({loss_l2.avg:.4f})\n'
                  'Loss l3 {loss_l3.val:.4f} ({loss_l3.avg:.4f})\n'
                  'Loss l4 {loss_l4.val:.4f} ({loss_l4.avg:.4f})\n'
                  'Accuracy l1 {acc_l1.val:.3f} ({acc_l1.avg:.3f})\n'
                  'Accuracy l2 {acc_l2.val:.3f} ({acc_l2.avg:.3f})\n'
                  'Accuracy l3 {acc_l3.val:.3f} ({acc_l3.avg:.3f})\n'
                  'Accuracy l4 {acc_l4.val:.3f} ({acc_l4.avg:.3f})\n'.format(i,
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, 
                                                                  loss_l1=losses_softmax[0],
                                                                  loss_l2=losses_softmax[1],
                                                                  loss_l3=losses_softmax[2],
                                                                  loss_l4=losses_softmax[3],
                                                                  acc_l1=accs_softmax[0],
                                                                  acc_l2 = accs_softmax[1],
                                                                  acc_l3 = accs_softmax[2],
                                                                  acc_l4 = accs_softmax[3]))

    print('Loss l1 ({loss_l1.avg:.4f})\t'
                  'Loss l2 ({loss_l2.avg:.4f})\n'
                  'Loss l3 ({loss_l3.avg:.4f})\n'
                  'Loss l4 ({loss_l4.avg:.4f})\n'
                  'Accuracy l1 ({acc_l1.avg:.3f})\n'
                  'Accuracy l2 ({acc_l2.avg:.3f})\n'
                  'Accuracy l3 ({acc_l3.avg:.3f})\n'
                  'Accuracy l4 ({acc_l4.avg:.3f})\n'.format(loss_l1=losses_softmax[0],
                                                                  loss_l2=losses_softmax[1],
                                                                  loss_l3=losses_softmax[2],
                                                                  loss_l4=losses_softmax[3],
                                                                  acc_l1=accs_softmax[0],
                                                                  acc_l2 = accs_softmax[1],
                                                                  acc_l3 = accs_softmax[2],
                                                                  acc_l4 = accs_softmax[3]))

    return accs_softmax[0].avg


if __name__ == '__main__':
    main()
