import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch_geometric.data import Data
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from model.cfd_error import EllipseAreaNetwork
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_ellipse_dataset(a, b, num_points, mesh_resolution):
    # Adjust the number of points based on mesh_resolution
    num_points_adjusted = int(num_points * mesh_resolution)

    # Generate points on ellipse boundary
    angles = np.linspace(0, 2 * np.pi, num_points_adjusted)
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    points = np.vstack((x, y)).T

    # Create mesh
    edge_index = np.array([(i, (i + 1) % num_points_adjusted) for i in range(num_points_adjusted)])
    reverse_edge_index = np.array([((i + 1) % num_points_adjusted, i) for i in range(num_points_adjusted)])
    edge_index = np.concatenate((edge_index, reverse_edge_index), axis=0)
    edge_lengths = np.linalg.norm(points[edge_index[:, 0]] - points[edge_index[:, 1]], axis=1)

    # Calculate the area of discretized elements
    base_lengths = edge_lengths[0:len(edge_lengths) // 2]
    # Calculate the length of the edge connecting the center of the ellipse to each point
    edges = np.sqrt(x ** 2 + y ** 2)

    target_edges = np.concatenate((edges[1:], edges[:1]))
    # Calculate the area by the Heron's formula
    p = (edges + target_edges + base_lengths) / 2
    areas = np.sqrt(p * (p - edges) * (p - target_edges) * (p - base_lengths)) 

    # Calculate the estimated area of the ellipse
    estimated_area = areas.sum()

    # Calculate the true area of the ellipse
    true_area = np.pi * a * b

    # Calculate the error between true area and estimated area
    y = true_area - estimated_area

    # Create PyG dataset
    data = Data(x=torch.tensor(points, dtype=torch.float),
                y=torch.tensor(y, dtype=torch.float).view(1, -1),
                # pos = torch.tensor(points, dtype=torch.float), 
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_lengths, dtype=torch.float))

    hyperparameters = torch.tensor([a, b, num_points, mesh_resolution], dtype=torch.float)

    return data, hyperparameters

def create_merged_ellipse_datasets(a_range, b_range, num_points, mesh_resolutions, num_samples):
    dataset = []
    hyperparameters = []
    # use tqdm to show the progress bar
    for _ in tqdm(range(num_samples)):
        for mesh_resolution in mesh_resolutions:
            # Randomly generate the ellipse parameters a and b within the specified ranges
            a = np.random.uniform(a_range[0], a_range[1])
            b = np.random.uniform(b_range[0], b_range[1])

            data, hyperparams = create_ellipse_dataset(a, b, num_points, mesh_resolution)
            dataset.append(data)
            hyperparameters.append(hyperparams)
    # save dataset and hyperparameters
    torch.save(dataset, "dataset/ellipse/dataset.pt")
    torch.save(hyperparameters, "dataset/ellipse/hyperparameters.pt")
    return dataset

def train_test_split(dataset, train_ratio):
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    return train_dataset, test_dataset

def visualize_alpha(writer, model, epoch):
    alphas = model.alpha[1]
    # alphas = np.array(alphas, dtype=np.float32)
    writer.add_histogram("Alpha", alphas, epoch)

def visualize_clusters(writer, data, model, epoch):
    clusters = model.cluster[1]
    # clusters = clusters.detach().cpu().numpy()

    fig = plt.figure()
    plt.scatter(data.x[:, 0].detach().cpu().numpy(), data.x[:, 1].detach().cpu().numpy(), c=clusters.detach().cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.title(f"Clusters (Epoch: {epoch})")

    writer.add_figure("Clusters", fig, epoch)

def main():
    # Parameters
    a_range = (3, 7)
    b_range = (1, 5)
    num_points = 40
    mesh_resolutions = [0.9, 0.8, 0.7, 0.6, 0.5]
    num_samples = 500
    batch_size = 32
    # in_channels = 2
    # out_channels = 1
    num_kernels = 4
    epochs = 500
    learning_rate = 0.001

    # Create datasets
    print("Creating datasets...")
    merged_dataset = create_merged_ellipse_datasets(a_range, b_range, num_points, mesh_resolutions, num_samples)

    # Split dataset
    train_dataset, test_dataset = train_test_split(merged_dataset, 0.8)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = EllipseAreaNetwork(num_kernels)
    model.to('cuda')

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    # Training loop
    model.train()
    error = []
    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to('cuda')
            optimizer.zero_grad()
            out = model(data)
            loss = torch.nn.MSELoss()(out, data.y)
            # reg_lambda = 1e-4  # Regularization weight
            # l2_reg = torch.norm(model.alpha, p=2)  # L2 regularization
            # loss = loss + reg_lambda * l2_reg
            abs_error = torch.abs(out - data.y)
            error.append(abs_error.cpu().detach().numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to('cuda')
                    out = model(data)
                    loss = torch.nn.MSELoss()(out, data.y)
                    test_loss += loss.item()

            test_loss /= len(test_loader)
            writer.add_scalar('Loss/test', test_loss, epoch)
            print(f"Test Loss: {test_loss}")
            # save the model
            torch.save(model.state_dict(), "model/ellipse/model.pt")

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
        writer.add_scalar('Loss/train', loss.item(), epoch)
        visualize_alpha(writer, model, epoch)
        visualize_clusters(writer, data, model, epoch)
    # concatenate the error list
    error = np.concatenate(error, axis=0)
    plt.plot(np.array(error, dtype=np.float32).squeeze())
    plt.show()


    writer.close()

if __name__ == "__main__":
    main()