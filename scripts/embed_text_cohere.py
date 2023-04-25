import os
import re
import pandas as pd
import configparser
import cohere
import time
from tqdm import tqdm
import argparse

def extract_links(text):
    links = re.findall(r"\[\[.*?\]\]", text)
    links = [link.strip("[[").strip("]]") for link in links]
    return links

def read_input_data(input_dir):
    data = []

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".md"):
                with open(os.path.join(root, filename), "r") as f:
                    text = f.read()

                links = extract_links(text)
                data.append({"filename": filename[:-3], "text": text, "links": links})

    df = pd.DataFrame(data, columns=["filename", "text", "links"])
    return df

def chunk_text(text, max_len=512):
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        if len(chunk) + len(word) + 1 <= max_len:
            chunk.append(word)
        else:
            chunks.append(' '.join(chunk))
            chunk = [word]

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

def create_embeddings(df, api_key, rate_limit_calls=100, rate_limit_duration=60, batch_size=96):
    co = cohere.Client(api_key)
    embeddings = []

    start_time = time.time()
    call_count = 0

    for i, row in df.iterrows():
        text_chunks = chunk_text(row['text'])

        for chunk in text_chunks:
            embeddings.append({
                'filename': row['filename'],
                'index': i,
                'chunk_text': chunk,
                'embedding': None
            })

    embedding_batches = [embeddings[i:i+batch_size] for i in range(0, len(embeddings), batch_size)]

    for batch in tqdm(embedding_batches, total=len(embedding_batches), desc="Embedding text"):
        texts = [item['chunk_text'] for item in batch]
        response = co.embed(texts=texts, model='large', truncate='END')

        for i, embedding in enumerate(response.embeddings):
            batch[i]['embedding'] = embedding

        call_count += 1

        if call_count >= rate_limit_calls:
            elapsed_time = time.time() - start_time

            if elapsed_time < rate_limit_duration:
                time.sleep(rate_limit_duration - elapsed_time)

            start_time = time.time()
            call_count = 0

    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings = df_embeddings.merge(df[['filename', 'links']], on='filename', how='left')
    return df_embeddings


def main():
    parser = argparse.ArgumentParser(description="Embed text from markdown files in a directory.")
    parser.add_argument("-i", "--input_dir", type=str, default="../input", help="Input directory containing markdown files.")
    parser.add_argument("-e", "--embeddings_csv", type=str, default="input_embeddings.csv", help="Output CSV file for embeddings dataframe.")
    parser.add_argument("-c", "--config", type=str, default="../config.ini", help="Configuration file containing the API key.")

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    api_key = config.get('cohere', 'api_key')

    # Read the input data
    df = read_input_data(args.input_dir)

    # Create embeddings
    df_embeddings = create_embeddings(df, api_key)
    df_embeddings.to_csv(args.embeddings_csv, index=False)

if __name__ == "__main__":
    main()