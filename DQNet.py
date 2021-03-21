import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from architecture import DGCNN_Net, MPNN

class DQNet():
    """
    Definition of the DQN: training, compute loss, predict, update model, etc.
    """
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DGCNN_Net(args.dimension).to(self.device)
        self.target_model = DGCNN_Net(args.dimension).to(self.device)
        #self.model = MPNN(args.dimension).to(self.device)
        #self.target_model = MPNN(args.dimension).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = args.learning_rate)

    def calc_loss_and_optimize(self, y_pred, y):
        """
        Compute the loss function and update the weights of network
        Return:
            Huber loss from y_pred and y_gt
        """
        self.model.train()
        loss = F.smooth_l1_loss(y_pred, y)
        #Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_with_grad(self, nn_input):
        self.model.train()
        y_pred = self.model(nn_input)
        return y_pred

    def predict(self, nn_input, target):
        """
        Predict Q value
        """
        with torch.no_grad():
            if target:
                self.target_model.eval()
                y_pred = self.target_model(nn_input)
            else:
                self.model.eval()
                y_pred = self.model(nn_input)
        return y_pred

    def update_target_model(self):
        """
        Update the target network, copying all weights and biases in DQN
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, folder, filename):
        """
        Save the model
        :param folder: Folder requested
        :param filename: file name requested
        """
        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.model.state_dict(), filepath)

    def load_saved_model(self, folder, filename):
        """
        Load the saved model
        folder: Folder requested
        filename: filename requested
        """
        filepath = os.path.join(folder, filename)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))