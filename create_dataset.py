import copy
import json
import os
import uuid

import networkx as nx
import pandas as pd
from bs4 import BeautifulSoup
from node2vec import Node2Vec
from tqdm import tqdm

FOLDER_PATH = "cleaneval/prepared_html/"
COUNTER = 0
SAVE_FILENAME = "result_v2.csv"

with open('tag_counts_cleaned.json') as f:
    data = json.load(f)
    TAGS_LIST = list(data.keys())


def save_result(result):
    # filename, node_name, node_text, graph embedding, parents embedding, words embedding, label
    try:
        df = pd.read_csv(SAVE_FILENAME)
    except:
        df = pd.DataFrame(columns=["filename", "node_name", "tags", "graph_embedding", "text", "label"])
    df_new = pd.DataFrame(result, columns=["filename", "node_name", "tags", "graph_embedding", "text", "label"])
    df_save = pd.concat([df, df_new])
    df_save.to_csv(SAVE_FILENAME, index=False)


def parse_tree(node, graph, tags_current, result, parent_name=None):
    global COUNTER
    unique_identifier = f"{str(node.name)}_{uuid.uuid4()}"

    if node.name == "span" and "__boilernet_label" in node.attrs:
        unique_identifier = f"{str(node.name)}_{COUNTER}"
        COUNTER += 1
        label = node.attrs['__boilernet_label']
        result.append([unique_identifier, copy.deepcopy(tags_current), [], node.text.strip(), label])
    if node.name in tags_current.keys():
        tags_current[node.name] += 1

    node_attrs = node.attrs if node.attrs else {}
    node_content = node.string.strip() if node.string else ''
    graph.add_node(unique_identifier, tag=node.name, attrs=node_attrs, content=node_content)

    if parent_name is not None:
        graph.add_edge(parent_name, unique_identifier)

    for child in node.children:
        if child.name is not None:
            parse_tree(child, graph, tags_current, result, parent_name=unique_identifier)


def process_article(path):
    global COUNTER
    COUNTER = 0
    with open(path, 'rb') as f:
        doc = BeautifulSoup(f, features='html5lib')
    graph = nx.DiGraph()
    current_tags = {tag: 0 for tag in TAGS_LIST}
    span_nodes_embeddings = []
    parse_tree(doc, graph, current_tags, span_nodes_embeddings)

    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    for row in span_nodes_embeddings:
        row[2] = model.wv[row[0]]

    return span_nodes_embeddings

    # plt.figure(1, figsize=(200, 80), dpi=60)
    # nx.draw(G, with_labels=True, pos=nx.spiral_layout(G))
    # plt.savefig("graph.png")


def create_dataset():
    paths_to_articles = sorted([os.path.join(FOLDER_PATH, file) for file in os.listdir(FOLDER_PATH)
                                if os.path.isfile(os.path.join(FOLDER_PATH, file))])
    result = []
    for path in tqdm(paths_to_articles):
        current_results = process_article(path)
        for row in current_results:
            result.append([path] + row)
        if len(result) > 1000:
            save_result(result)
            result = []
    save_result(result)


if __name__ == "__main__":
    create_dataset()
