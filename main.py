import sys
import getopt
from model import train_ANN

def message():
    print 'main.py --ms <memory_size> \
                   --hs <hidden_size>   \
                   --lr <learning_rate> \
                   --ep <epochs_of_training> \
                   --help'

def main(argv):
    try:
        opts, args = getopt.getopt(argv,'',["ms=","hs=","lr=","ep=","help="])
    except getopt.GetoptError:
        print "Training with default parameters"
        train_ANN()
        message()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--ms":
            mem_size = int(arg)
        elif opt == "--hs":
            hidden_size = int (arg)
        elif opt == "--lr":
            learning_rate = float(arg)
        elif opt == "--ep":
            epochs = int(arg)
        elif opt == "--help":
            message()

    if len(sys.argv) == 9:
        train_ANN(mem_size=mem_size,
                   hidden_size=hidden_size,
                   learning_rate=learning_rate,
                   epochs=epochs)
    else:
        print "Missing parameters, please insert all"
        message()

if __name__ == "__main__":
    main(sys.argv[1:])
