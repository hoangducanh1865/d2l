import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

d2l.use_svg_display()

# kernels
def gaussian(x): return torch.exp(-x**2 / 2)
def boxcar(x): return (torch.abs(x) < 1.0).to(x.dtype)
def constant(x): return 1.0 + 0 * x
def epanechikov(x): return torch.max(1 - torch.abs(x), torch.zeros_like(x))

def f(x):
    return 2 * torch.sin(x) + x

def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = x_train.reshape((-1, 1)) - x_val.reshape((1, -1))  # (n_train, n_val)
    k = kernel(dists).type(torch.float32)                     # (n_train, n_val)
    attention_w = k / (k.sum(0, keepdim=True) + 1e-12)        # normalize columns; add eps for stability
    y_hat = y_train @ attention_w                             # (n_val,)
    return y_hat, attention_w

def plot_results(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, len(kernels), sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            pcm = ax.imshow(attention_w.detach().numpy(), aspect='auto')
        else:
            ax.plot(x_val.numpy(), y_hat.numpy(), label='y_hat')
            ax.plot(x_val.numpy(), y_val.numpy(), 'm--', label='y_true')
            ax.plot(x_train.numpy(), y_train.numpy(), 'o', alpha=0.5, label='train')
            ax.legend()
        ax.set_xlabel(name)
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
    return fig

def main():
    kernels = (gaussian, boxcar, constant, epanechikov)
    names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')

    # Plot kernels
    x = torch.arange(-2.5, 2.5, 0.1)
    fig1, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        ax.plot(x.numpy(), kernel(x).numpy())
        ax.set_xlabel(name)
    path1 = "/kaggle/working/kernels.png"
    fig1.savefig(path1)
    plt.close(fig1)
    print("Saved", path1)

    # Generate data
    n = 40
    x_train, _ = torch.sort(torch.rand(n) * 5)
    y_train = f(x_train) + torch.randn(n)
    x_val = torch.arange(0, 5, 0.1)
    y_val = f(x_val)

    # Plot predictions
    fig2 = plot_results(x_train, y_train, x_val, y_val, kernels, names, attention=False)
    path2 = "/kaggle/working/predictions.png"
    fig2.savefig(path2)
    plt.close(fig2)
    print("Saved", path2)

    # Plot attention heatmaps
    fig3 = plot_results(x_train, y_train, x_val, y_val, kernels, names, attention=True)
    path3 = "/kaggle/working/attentions.png"
    fig3.savefig(path3)
    plt.close(fig3)
    print("Saved", path3)

if __name__ == "__main__":
    main()