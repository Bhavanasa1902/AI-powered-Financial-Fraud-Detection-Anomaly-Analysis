import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TransactionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 16, 8]):
        super(TransactionAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            encoder_layers.append(nn.Linear(in_d, h_d))
            encoder_layers.append(nn.ReLU())
            in_d = h_d
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for h_d in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(in_d, h_d))
            decoder_layers.append(nn.ReLU())
            in_d = h_d
            
        decoder_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DeepAnomalyDetector:
    def __init__(self, input_dim, epochs=10, batch_size=256, lr=1e-3):
        self.model = TransactionAutoencoder(input_dim)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.threshold = None

    def train(self, X_train):
        print(f"Training Autoencoder on {self.device}...")
        self.model.train()
        dataset = TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                inputs = batch[0].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.6f}")

        # Determine reconstruction error threshold based on 95th percentile of training data
        errors = self.get_reconstruction_error(X_train)
        self.threshold = np.percentile(errors, 95)
        print(f"Reconstruction Error 95th Percentile Threshold: {self.threshold:.4f}")

    def get_reconstruction_error(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # MSE per sample
            errors = torch.mean((X_tensor - outputs)**2, dim=1).cpu().numpy()
        return errors

    def predict_anomaly_score(self, X):
        # Scale to 0-1 based loosely on threshold
        errors = self.get_reconstruction_error(X)
        scores = np.clip(errors / (self.threshold * 2), 0, 1)
        return scores

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold
        }, path)
        print(f"Autoencoder saved to {path}")

    def load(self, path, input_dim):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model = TransactionAutoencoder(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
