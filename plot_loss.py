import pickle
import numpy
import matplotlib.pyplot as plt
import os
from config import Config
config = Config()

os.makedirs(f"plots/{config.model_name}", exist_ok=True)

def plot_loss(epoch, all_reconloss, all_ncut):

    number_iter = 100
    recon = numpy.array(all_reconloss) # changing to array
    recon = recon.reshape((-1,1)) #making a single column
    div = int(recon.shape[0]/number_iter) # getting the last iteration divisible by 100
    recon = recon[:div*number_iter]
    recon = recon.reshape((-1,number_iter)).mean(axis=1) # takin mean of 100 iterations

    n_cut = numpy.array(all_ncut)
    n_cut = n_cut.reshape((-1,1))
    div = int(n_cut.shape[0]/number_iter)
    n_cut = n_cut[:div*number_iter]
    n_cut = n_cut .reshape((-1,number_iter)).mean(axis=1)

    plt.figure(figsize=(10,5))
    plt.suptitle(f"Epoch {epoch+1}")
    # plt.figure(1)
    plt.subplot(121)
    plt.title("Reconstruction Loss")
    plt.ylabel("Loss")
    plt.xlabel(f"Per {number_iter} Iteration (Batch Size {config.batch_size})")
    plt.plot(recon)

    # plt.figure(2)
    plt.subplot(122)
    plt.tight_layout()
    plt.title("N-Cut Loss")
    plt.ylabel("Loss")
    plt.xlabel(f"Per {number_iter} Iteration (Batch Size {config.batch_size})")
    plt.plot(n_cut)
    # plt.show()

    plt.savefig(f"plots/{config.model_name}/{epoch}")
    plt.clf()

if __name__ == "__main__":

    print("Plotting...")

    all_reconloss = []
    all_ncut = []

    with open('reconstruction_loss.pkl', 'rb') as f:
        while True:
            try:
                all_reconloss.append(pickle.load(f))
            except EOFError:
                break

    with open('n_cut_loss.pkl', 'rb') as fb:
        while True:
            try:
                all_ncut.append(pickle.load(fb))
            except EOFError:
                break
    plot_loss("last_epoch", all_reconloss, all_ncut)
