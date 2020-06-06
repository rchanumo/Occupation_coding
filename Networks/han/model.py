import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class HierarchialAttentionNetwork(nn.Module):
    """
    Hierarchial Attention Network (HAN).
    """

    def __init__(self, n_classes, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers,
                 sentence_rnn_layers, word_att_size, sentence_att_size, bert_feature_dim=0, dropout=0.5, attn=1):

        super(HierarchialAttentionNetwork, self).__init__()

        # Sentence-level attention module (which will, in-turn, contain the word-level attention module)
        self.sentence_attention = SentenceAttention(vocab_size, emb_size, word_rnn_size, sentence_rnn_size,
                                                    word_rnn_layers, sentence_rnn_layers, word_att_size,
                                                    sentence_att_size, bert_feature_dim, dropout, attn)

        # Classifier
        self.fc = nn.Linear(2 * sentence_rnn_size, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence, bert_features=None):

        # Apply sentence-level attention module (and in turn, word-level attention module) to get document embeddings
        document_embeddings, word_alphas, sentence_alphas = self.sentence_attention(documents, sentences_per_document,
                                                                                    words_per_sentence, bert_features) 

        # Classify
        document_embeddings = self.dropout(document_embeddings)
        scores = self.fc(document_embeddings)

        return document_embeddings, scores, word_alphas, sentence_alphas


class SentenceAttention(nn.Module):
    """
    The sentence-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers,
                 word_att_size, sentence_att_size, bert_feature_dim, dropout, attn):
    
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size,
                                            dropout, attn)

        # Bidirectional sentence-level RNN
        self.sentence_rnn = nn.GRU(2*word_rnn_size+bert_feature_dim, sentence_rnn_size, num_layers=sentence_rnn_layers,
                                   bidirectional=True, dropout=dropout, batch_first=True)

        # Sentence-level attention network
        self.sentence_attention = nn.Linear(2 * sentence_rnn_size, sentence_att_size)

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(sentence_att_size, 1,
                                                 bias=False)  # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector

        # Dropout
        self.dropout = nn.Dropout(dropout)

        #apply sentence attenstion or not
        self.attn = attn

    def forward(self, documents, sentences_per_document, words_per_sentence, bert_features):
  
        # Sort documents by decreasing document lengths (SORTING #1)
        sentences_per_document, doc_sort_ind = sentences_per_document.sort(dim=0, descending=True)
        documents = documents[doc_sort_ind] 
        words_per_sentence = words_per_sentence[doc_sort_ind] 

        # Re-arrange as sentences by removing pad-sentences (DOCUMENTS -> SENTENCES)
        sentences, bs = pack_padded_sequence(documents,
                                             lengths=sentences_per_document.tolist(),
                                             batch_first=True)  
        # Re-arrange sentence lengths in the same way (DOCUMENTS -> SENTENCES)
        words_per_sentence, _ = pack_padded_sequence(words_per_sentence,
                                                     lengths=sentences_per_document.tolist(),
                                                     batch_first=True) 

        # Find sentence embeddings by applying the word-level attention module
        sentences, word_alphas = self.word_attention(sentences,
                                                     words_per_sentence) 
        sentences = self.dropout(sentences)

        sentences, _ = pad_packed_sequence(PackedSequence(sentences, bs),
                                       batch_first=True)
        if(not bert_features is None):
            sentences = torch.cat((sentences, bert_features) , 2)

        sentences, _ = pack_padded_sequence(sentences,
                                        lengths=sentences_per_document.tolist(),
                                        batch_first=True)

        # Apply the sentence-level RNN over the sentence embeddings 
        (sentences, _), _ = self.sentence_rnn(
            PackedSequence(sentences, bs)) 

        # Find attention vectors by applying the attention linear layer
        att_s = self.sentence_attention(sentences) 
        att_s = F.tanh(att_s)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s).squeeze(1)

        max_value = att_s.max() 
        att_s = torch.exp(att_s - max_value)

        # Re-arrange as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        att_s, _ = pad_packed_sequence(PackedSequence(att_s, bs),
                                       batch_first=True)  

        # Calculate softmax values
        sentence_alphas = att_s / torch.sum(att_s, dim=1, keepdim=True)

        # Similarly re-arrange sentence-level RNN outputs as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        documents, _ = pad_packed_sequence(PackedSequence(sentences, bs),
                                           batch_first=True)

        # Find document embeddings
        if(self.attn):
            documents = documents * sentence_alphas.unsqueeze(
                2)
        documents = documents.sum(dim=1)

        # Also re-arrange word_alphas (SENTENCES -> DOCUMENTS)
        word_alphas, _ = pad_packed_sequence(PackedSequence(word_alphas, bs),
                                             batch_first=True)

        # Unsort documents into the original order (INVERSE OF SORTING #1)
        _, doc_unsort_ind = doc_sort_ind.sort(dim=0, descending=False) 
        documents = documents[doc_unsort_ind]
        sentence_alphas = sentence_alphas[doc_unsort_ind]
        word_alphas = word_alphas[doc_unsort_ind]

        return documents, word_alphas, sentence_alphas


class WordAttention(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout, attn):
        
        super(WordAttention, self).__init__()

        # Embeddings (look-up) layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # Bidirectional word-level RNN
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
                               dropout=dropout, batch_first=True)

        # Word-level attention network
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

        #apply word attenstion or not
        self.attn = attn

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer?
        """
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune

    def forward(self, sentences, words_per_sentence):
        
        # Sort sentences by decreasing sentence lengths (SORTING #2)
        words_per_sentence, sent_sort_ind = words_per_sentence.sort(dim=0, descending=True)
        sentences = sentences[sent_sort_ind]  

        # Get word embeddings, apply dropout
        sentences = self.dropout(self.embeddings(sentences)) 

        # Re-arrange as words by removing pad-words (SENTENCES -> WORDS)
        words, bw = pack_padded_sequence(sentences,
                                         lengths=words_per_sentence.tolist(),
                                         batch_first=True) 

        # Apply the word-level RNN over the word embeddings 
        (words, _), _ = self.word_rnn(PackedSequence(words, bw))  

        # Find attention vectors by applying the attention linear layer
        att_w = self.word_attention(words) 
        att_w = F.tanh(att_w) 
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1) 

        # First, take the exponent
        max_value = att_w.max()  
        att_w = torch.exp(att_w - max_value)  

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(PackedSequence(att_w, bw),
                                       batch_first=True) 

        # Calculate softmax values
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(PackedSequence(words, bw),
                                           batch_first=True)  

        # Find sentence embeddings
        if(self.attn):
            sentences = sentences * word_alphas.unsqueeze(2)  
        sentences = sentences.sum(dim=1) 

        # Unsort sentences into the original order (INVERSE OF SORTING #2)
        _, sent_unsort_ind = sent_sort_ind.sort(dim=0, descending=False)  
        sentences = sentences[sent_unsort_ind] 
        word_alphas = word_alphas[sent_unsort_ind]

        return sentences, word_alphas
