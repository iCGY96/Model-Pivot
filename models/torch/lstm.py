import torch
import torch.nn as nn

SEQ_LEN = 3
BATCH_SIZE = 1
IMAGE_SIZE = 224
INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE
HIDDEN_DIM = 128


class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE,
                            hidden_size=HIDDEN_DIM,
                            num_layers=1)

    def forward(self, data_input):
        data_input = torch.reshape(data_input,
                                   (SEQ_LEN, BATCH_SIZE, INPUT_SIZE))

        lstm_output, (hidden_state, cell_state) = self.lstm(data_input)
        return lstm_output


def main():
    x = torch.zeros(BATCH_SIZE, SEQ_LEN, IMAGE_SIZE, IMAGE_SIZE)
    net = Lstm()
    output = net(x)
    print(output.shape)
    print(output)


if __name__ == '__main__':
    main()
