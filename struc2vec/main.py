#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse, logging
import numpy as np
from .struc2vec import Graph
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time
import networkx as nx

from . graph import from_numpy


logging.basicConfig(filename='struc2vec.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')

def parse_args():
    '''
    Parses the struc2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run struc2vec.")

    parser.add_argument('--input', nargs='?', default='../graph/karate-mirrored.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='../emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--until-layer', type=int, default=None,
                        help='Calculation until the layer.')

    parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--OPT1', default=False, type=bool,
                      help='optimization 1')
    parser.add_argument('--OPT2', default=False, type=bool,
                      help='optimization 2')
    parser.add_argument('--OPT3', default=False, type=bool,
                      help='optimization 3')    
    return parser.parse_args()


def read_graph(A, directed):
    '''
    Reads the input network.
    '''
    logging.info(" - Loading graph...")
    G = from_numpy(A, undirected= not directed)
    logging.info(" - Graph loaded.")
    return G


def learn_embeddings(output, d, window_size, workers, iter):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    logging.info("Initializing creation of the representations...")
    walks = LineSentence('random_walks.txt')
    model = Word2Vec(walks, size=d, window=window_size, min_count=0, hs=1, sg=1, workers=workers, iter=iter)
    model.wv.save_word2vec_format(output)
    res = {}
    with open(output) as inp_fifle:
        for i, line in enumerate(inp_fifle):
            if i < 1:
                continue
            fields = [float(v) for v in line.split(' ')]
            node_id = int(fields[0])
            emb = fields[1:]
            res[node_id] = emb
    embds = np.array([res[k] for k in sorted(res)])
    logging.info("Representations created.")
    
    return embds


def exec_struc2vec(A,
                   directed=True,
                   workers=8,
                   num_walks=10,
                   walk_length=80,
                   output='embeddings.txt',
                   opt1=True,
                   opt2=True,
                   opt3=True,
                   ul=None):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if(opt3):
        until_layer = ul
    else:
        until_layer = None

    G = read_graph(A, directed)
    G = Graph(G, directed, workers, untilLayer=until_layer)

    if(opt1):
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()

    if(opt2):
        G.create_vectors()
        G.calc_distances(compactDegree=opt1)
    else:
        G.calc_distances_all_vertices(compactDegree=opt1)

    G.create_distances_network()
    G.preprocess_parameters_random_walk()

    G.simulate_walks(num_walks, walk_length)

    return G


def prepare_embeddings(A, d=16, output='embds.txt',  window_size=10, workers=8, iters=10):
    exec_struc2vec(A, workers=workers)

    embeddings = learn_embeddings(output, d, window_size, workers, iters)

    return embeddings


if __name__ == '__main__':
    G = nx.read_edgelist('../graph/karate-mirrored.edgelist')
    prepare_embeddings(nx.adjacency_matrix(G))
