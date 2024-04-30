import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from strnn.models.strNN import StrNN
from strnn.models.model_utils import NONLINEARITIES

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class DataBundle:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def load_data(dataset_name='CiteSeer', transform=NormalizeFeatures()):
    print("Loading data...")
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name, transform=transform)
    data = dataset[0].to(device)
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    print("Adjacency matrix created.")

    num_classes = dataset.num_classes
    return data, adj_matrix.cpu().numpy(), num_nodes, num_classes


def evaluate_model(feature_extractor, classifier, embedding_layer, data, criterion):
    print("Evaluating model...")
    feature_extractor.eval()
    classifier.eval()
    embedding_layer.eval()  

    with torch.no_grad():
        # Ensure that data.x is passed through the embedding layer first.
        embedded_x = embedding_layer(data.x).to(device)
        features = feature_extractor(embedded_x)
        out = classifier(features)
        loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
        pred = out.argmax(dim=1)
        correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
        accuracy = correct / int(data.val_mask.sum())

    print(f"Validation Loss: {loss}, Accuracy: {accuracy}")
    return loss, accuracy


def train_loop(feature_extractor, classifier, optimizer, criterion, train_data, val_data, max_epoch, patience):
    print("Starting training...")
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0

    for epoch in range(max_epoch):
        print(f"Epoch {epoch + 1}/{max_epoch}")

        # Training phase
        feature_extractor.train()
        classifier.train()
        optimizer.zero_grad()
        features = feature_extractor(train_data.x)
        predictions = classifier(features)
        loss = criterion(predictions, train_data.y)
        loss.backward()
        optimizer.step()

        # Detach train_data.x after update to prevent accumulation of gradient history
        train_data.x = train_data.x.detach()

        feature_extractor.eval()
        classifier.eval()
        with torch.no_grad():  
            val_out = classifier(feature_extractor(val_data.x))
            val_loss = criterion(val_out, val_data.y).item()

        print(f"Training Loss: {loss.item()}, Validation Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'feature_extractor': feature_extractor.state_dict(),
                'classifier': classifier.state_dict()
            }
            counter = 0
            print("New best model saved.")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {patience} epochs with no improvement.")
                break

    feature_extractor.load_state_dict(best_model_state['feature_extractor'])
    classifier.load_state_dict(best_model_state['classifier'])

    return best_val_loss


def main():
    data, adj_matrix, num_nodes, num_classes = load_data('CiteSeer')
    print("Data loaded successfully.")

    embedding_layer = FeatureEmbedding(data.num_features, num_nodes).to(device)

    # Prepare training and validation data bundles
    train_x_embedded = embedding_layer(data.x[data.train_mask]).to(device)
    train_y = data.y[data.train_mask].to(device)
    val_x_embedded = embedding_layer(data.x[data.val_mask]).to(device)
    val_y = data.y[data.val_mask].to(device)

    train_data = DataBundle(train_x_embedded, train_y)
    val_data = DataBundle(val_x_embedded, val_y)

    print("Setting up feature extractor and classifier...")
    feature_extractor = StrNN(nin=num_nodes, hidden_sizes=[num_nodes * 2], nout=num_nodes, opt_type='greedy',
                              adjacency=adj_matrix, init=1, activation='relu').to(device)

    classifier = nn.Linear(num_nodes, num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters()) + list(classifier.parameters()) + list(embedding_layer.parameters()),
        lr=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_loss = train_loop(feature_extractor, classifier, optimizer, criterion, train_data, val_data, 100, 10)
    print(f"Best validation loss: {best_val_loss}")

    test_loss, test_accuracy = evaluate_model(feature_extractor, classifier, embedding_layer, data, criterion)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()