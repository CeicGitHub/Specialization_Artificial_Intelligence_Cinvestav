
#* CESAR EDUARDO INDA CENICEROS

#! Resumen_Funcionalidad:
''' este código implementa un sistema de clasificación de imágenes de prendas utilizando el conjunto de datos FashionMNIST y una red neuronal de tipo MLP. 
--> Primero, descarga y prepara las imágenes, selecciona una muestra de 3,000 ejemplos y la divide en datos de entrenamiento y validación. 
--> Cada imagen, originalmente de 28×28 píxeles, se convierte en un vector de 784 valores para poder ser procesada por la red neuronal.
--> La arquitectura utilizada contiene dos capas ocultas de 512 y 256 neuronas con la función de activación ReLU, 
--> además de una capa de salida con 10 neuronas, una por cada categoría de ropa.
--> comprobar si la combinación de múltiples redes neuronales puede obtener una clasificación más precisa y estable que la obtenida por cada modelo individual.
--> reducir los errores particulares de cada red y mejorar la capacidad de generalización del sistema ante imágenes que no fueron utilizadas durante el entrenamiento.
'''

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms
    from torch.utils.tensorboard import SummaryWriter
    import math
    import marimo as mo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    return datasets, device, nn, plt, torch, transforms


@app.cell
def _(datasets, transforms):
    full_train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    len(full_train_dataset)
    return (full_train_dataset,)


@app.cell
def _(datasets, transforms):
    test_datset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    len(test_datset)
    return (test_datset,)


@app.cell
def _():
    id_label_map = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    return (id_label_map,)


@app.cell
def _(id_label_map, plt):
    def plot_sample(data, label):
        plt.figure(figsize=(2, 2))
        plt.imshow(data.squeeze(), cmap="gray")
        plt.title(f"{id_label_map[label]}")
        plt.show()

    return


@app.cell
def _(full_train_dataset, torch):
    torch.manual_seed(42)

    N = 3000

    indices = torch.randperm(len(full_train_dataset))[:N]
    indices
    return (indices,)


@app.cell
def _(full_train_dataset, indices, torch):
    small_dataset = torch.utils.data.Subset(full_train_dataset, indices)
    len(small_dataset)
    return (small_dataset,)


@app.cell
def _():
    # _idx = 1000
    # plot_sample(small_dataset[_idx][0], small_dataset[_idx][1])
    return


@app.cell
def _(small_dataset, torch):
    train_dataset, val_dataset = torch.utils.data.random_split(
        small_dataset,
        [2400, 600],
        generator=torch.Generator().manual_seed(42)
    )
    len(train_dataset), len(val_dataset)
    return train_dataset, val_dataset


@app.cell
def _(test_datset, torch, train_dataset, val_dataset):
    batch_size = 64

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_datset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    len(train_loader), len(val_loader), len(test_loader)
    return (test_loader,)


#! Esta clase si incluye "Dropout"
@app.cell
def _(nn):
    class MLP(nn.Module):
        def __init__(self, dropout_rate=0.2):
            super(MLP, self).__init__()

            self.nn = nn.Sequential(
                nn.Flatten(),

                nn.Linear(28 * 28, 512),  # nn.1
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                nn.Linear(512, 256),      # nn.4
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                nn.Linear(256, 10)        # nn.7
            )

        def forward(self, x):
            return self.nn(x)

    return (MLP,)


""""" 
@app.cell
def _(nn):
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()

            self.nn = nn.Sequential(
                nn.Flatten(),

                nn.Linear(28 * 28, 512), # Layer 1
                nn.ReLU(),

                nn.Linear(512, 256), # Layer 2
                nn.ReLU(),

                nn.Linear(256, 10) # Output layer
            )

        def forward(self, x):
            return self.nn(x)

    return (MLP,)
"""

@app.cell
def _(device, torch):
    @torch.no_grad()
    def evaluate(model, loss_fn, loader):
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            total_loss += loss.item()
            preds = torch.argmax(y_pred, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return total_loss / len(loader), correct / total

    return (evaluate,)


@app.cell
def _(MLP, device, evaluate, nn, test_loader, torch):
    loss_fn = nn.CrossEntropyLoss()

    model_dirs = [
        "runs/l2_reg_0.0001_patience_10_epochs_100_dropout_0.2/model.pth",
        "runs/l2_reg_0.001_patience_10_epochs_100_dropout_0.2/model.pth",
        "runs/l2_reg_0.001_patience_15_epochs_100_dropout_0.3/model.pth",
        "runs/l2_reg_0.001_patience_20_epochs_100_dropout_0.1/model.pth",
        "runs/l2_reg_0.01_patience_10_epochs_100_dropout_0.2/model.pth",
        #"runs/l2_reg_0.001_patience_5_epochs_500/model.pth",
        #"runs/l2_reg_0.0001_patience_20_epochs_500/model.pth",
        #"runs/l2_reg_0.1_patience_20_epochs_500/model.pth",
        #"runs/l2_reg_0.001_patience_10_epochs_500/model.pth",
        #"runs/l2_reg_0.001_patience_15_epochs_500/model.pth"
    ]
    models = []

    for _i, model_dir in enumerate(model_dirs):
        model = MLP().to(device)
        model.load_state_dict(torch.load(model_dir, weights_only=False))
        models.append(model)

        loss, accuracy = evaluate(model, loss_fn, test_loader)

        print(f"[{_i}] -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return (models,)


@app.cell
def _(device, models, test_loader, torch):
    for _model in models:
        _model.eval()

    _correct = 0
    _total = 0

    with torch.no_grad():
        for _x, _y in test_loader:
            _x = _x.to(device)
            _y = _y.to(device)

            _predictions = []

            for _model in models:
                _y_pred = _model(_x)
                _predictions.append(_y_pred)
        
        
            _stacked_predictions = torch.stack(_predictions, dim=0)
        
            # To compute different ensembles change this!. 
            # Compute mean
            _mean_predictions = torch.mean(_stacked_predictions, dim=0)

            # Compute median
            # _median_predictions = torch.median(_stacked_predictions, dim=0).values

            _predictions = _mean_predictions.argmax(dim=1)

            _correct += (_predictions == _y).sum().item()
            _total += _y.size(0)
    
        _emsemble_accuracy = _correct / _total
            
    print(f"Ensemble Accuracy: {_emsemble_accuracy:.4f}")
    return


@app.cell
def _():
    return

if __name__ == "__main__":
    app.run()
