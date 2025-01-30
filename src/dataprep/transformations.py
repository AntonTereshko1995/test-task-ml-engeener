import pandas as pd
import re
import torch
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

def remove_empty_rows(data, columns=None):
    if columns:
        return data.dropna(subset=columns)
    return data.dropna()

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def set_low_register(data_frame):
    return data_frame.applymap(lambda x: x.lower() if isinstance(x, str) else x)

def combine_columns(df, columns):
    return df[columns].apply(lambda x: [label for label in x if label], axis=1)

def get_unique_labels(df, columns):
    return set(df[columns].values.flatten()) - {""}

def encode_data(row, data):
    return [1 if label in row.values else 0 for label in data]

def apply_one_hot_encoding(df, columns):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[columns].astype(str))
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns))
    return encoded_df

def apply_bert_embeddings(df, columns):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    combined_text = df[columns].apply(lambda x: ' '.join(x), axis=1)
    embeddings = combined_text.apply(get_bert_embedding)
    embeddings_df = pd.DataFrame(embeddings.tolist())
    return embeddings_df

def apply_tfidf_vectorization(df, text_column, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)  
    tfidf_matrix = vectorizer.fit_transform(df[text_column].astype(str).fillna(""))
    return tfidf_matrix

def apply_multilabel_binarization(df, column):
    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(df[column])
    return binary_matrix