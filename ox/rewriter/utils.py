from ox.rewriter.rewriter import UnitRewriterBase
from ox.tensorflow.rewriter.gru_rewriter import GRURewriter
from ox.tensorflow.rewriter.lstm_rewriter import LSTMRewriter

def process_graph(graph, weights):
    rewriter_list = [GRURewriter, LSTMRewriter]

    for rewriter in rewriter_list:
        rewriter(graph, weights).run()