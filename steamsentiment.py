import pandas as pd
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
from string import punctuation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import spacy
from sklearn.model_selection import train_test_split

print("Loading dataset")

# Load the data
data = pd.read_csv('game/steam.csv')

print("Dataset loaded")

# Define Fields
spacy_en = spacy.load("en_core_web_sm")
TEXT = Field(tokenize=lambda text: [tok.text for tok in spacy_en.tokenizer(text)], lower=True, include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

# Preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""

    no_punctuation = text.translate(str.maketrans('', '', punctuation))
    tokens = [tok.text for tok in spacy_en.tokenizer(no_punctuation)]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

data['review_text'] = data['review_text'].apply(preprocess_text)

# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Convert to TabularDataset
fields = [('review_text', TEXT), ('review_score', LABEL)]
train_dataset = TabularDataset(dataframe=train_data, format='csv', fields=fields, skip_header=True)
valid_dataset = TabularDataset(dataframe=valid_data, format='csv', fields=fields, skip_header=True)
test_dataset = TabularDataset(dataframe=test_data, format='csv', fields=fields, skip_header=True)

print("Loading Vocab")

# Build vocabulary
TEXT.build_vocab(train_dataset, vectors="glove.twitter.27B.50d.txt")

# Define DataLoader
batch_size = 50
train_loader, valid_loader, test_loader = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_sizes=(batch_size, batch_size, batch_size),
    sort_key=lambda x: len(x.review_text),
    device=None,  # Use GPU if available
    sort_within_batch=True,
    repeat=False
)

dataiter = iter(train_loader)

sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.review_text.size())
print('Sample input: \n', sample_x.review)
print('Sample label size: ', sample_y.review_text.size())
print('Sample label: \n', sample_y)

class SentimentalLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        print("Initializing Sentimental LSTM Model\n")

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(hidden_dim, output_size)  # Fix the input size here
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        
        print("Forwarding LSTM Model\n")

        embedd = self.embedding(x)
        lstm_out, hidden = self.lstm(embedd, hidden)

        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        sig_out = self.sigmoid(out)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

vocab_size = len(TEXT.vocab)
output_size = 1
embedding_dim = 50
hidden_dim = 256
n_layers = 2

net = SentimentalLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# Check if GPU is available and move the model to GPU
if torch.cuda.is_available():
    net.cuda()
    
print(net)
