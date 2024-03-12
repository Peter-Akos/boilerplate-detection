import pandas as pd
import tensorflow_hub as hub
from tqdm import tqdm


def generate_sentence_embeddings(input_filename, output_filename):
    df = pd.read_csv(input_filename)
    # Load the Universal Sentence Encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    words_to_embed = set(df['text'].to_list())

    history = {}

    for word in tqdm(words_to_embed[:20]):
        history[word] = embed(word)

    embedded_texts = []

    for word in df['text'].to_list():
        embedded_texts.append(history[word])

    df['embedded_texts'] = embedded_texts

    df.to_csv(output_filename)


if __name__ == "__main__":
    generate_sentence_embeddings("result_v2.csv", "final_data_v2.csv")
