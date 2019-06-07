
from common.rewriter.rewriter import UnitRewriterBase
from common.tensorflow.rewriter.gru_rewriter import GRURewriter
from common.tensorflow.rewriter.lstm_rewriter import LSTMRewriter

def process_graph(graph, weights):
    rewriter_list = [GRURewriter, LSTMRewriter]

    for rewriter in rewriter_list:
        rewriter(graph, weights).run()