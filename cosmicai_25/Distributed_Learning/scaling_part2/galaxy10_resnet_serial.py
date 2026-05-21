import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image



import os

with h5py.File('/tmp/data/Galaxy10_DECals.h5','r') as File:
    labels = np.array(File['ans'])
    indSub = np.where((labels==2) | (labels==5) | (labels==8))
    images = np.array(File['images'])[indSub]
    labels = labels[indSub]
    d = {2:0, 5:1, 8:2}
    mp = np.arange(0,9)
    mp[[int(k) for k in d.keys()]] = [[int(v) for v in d.values()]]
    labels = mp[labels]

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
def load_datasets():
  data_train, data_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.2, random_state=42)
  #  Main Modification: Additional transformation
  transform_train = transforms.Compose([transforms.AutoAugment(),  
                                transforms.Resize((224,224)),
                                transforms.ToTensor()])
  train_dataset = MyDataset(data_train, y_train, transform=transform_train)
                                      
  transform_val = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor()])                                     
  val_dataset = MyDataset(data_valid, y_valid, transform=transform_val)
  print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

  return train_dataset, val_dataset

def construct_dataloaders(train_set, val_set, batch_size, shuffle=True):
  train_dataloader = torch.utils.data.DataLoader(train_set, batch_size, shuffle)
  val_dataloader = torch.utils.data.DataLoader(val_set, batch_size) 
  return train_dataloader, val_dataloader


def getResNet():
  resnet = models.resnet34(weights='IMAGENET1K_V1')

  # Fix the conv layers parameters
  for conv_param in resnet.parameters():
    conv_param.require_grad = False

  # get the input dimension for this layer
  num_ftrs = resnet.fc.in_features
    
  # build the new final mlp layers of network
  fc = nn.Sequential(
          nn.Linear(num_ftrs, num_ftrs),
          nn.ReLU(),
          nn.Linear(num_ftrs, 3)
        )
    
  # replace final fully connected layer
  resnet.fc = fc
  return resnet

def load_checkpoint(checkpoint_path, DEVICE):
  checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
  return checkpoint

@torch.no_grad()
def eval_model(data_loader, model, loss_fn, DEVICE):
  model.eval()
  loss, accuracy = 0.0, 0.0
  n = len(data_loader)

  for i, data in enumerate(data_loader):
    x,y = data
    x,y = x.to(DEVICE), y.to(DEVICE)
    pred = model(x)
    loss += loss_fn(pred, y)/len(x)
    pred_label = torch.argmax(pred, axis = 1)
    accuracy += torch.sum(pred_label == y)/len(x)

  return loss/n, accuracy/n 

def train(train_loader, val_loader, model, opt, scheduler, loss_fn, epochs, DEVICE, checkpoint_file, prev_best_val_acc):
  n = len(train_loader)
  
  best_val_acc = torch.tensor(0.0).cuda() if prev_best_val_acc is None else prev_best_val_acc
    
  for epoch in range(epochs):
    model.train(True)
    
    avg_loss, val_loss, val_acc, avg_acc  = 0.0, 0.0, 0.0, 0.0
    
    start_time = datetime.now()
    
    for x, y in train_loader:
      x, y = x.to(DEVICE), y.to(DEVICE)
      pred = model(x)
      loss = loss_fn(pred,y)

      opt.zero_grad()
      loss.backward()
      opt.step()

      avg_loss += loss.item()/len(x)
      pred_label = torch.argmax(pred, axis=1)
      avg_acc += torch.sum(pred_label == y)/len(x)

    val_loss, val_acc = eval_model(val_loader, model, loss_fn, DEVICE)
    
    end_time = datetime.now()
    
    total_time = torch.tensor((end_time-start_time).seconds).cuda()
    
    #######################################
    # Learning rate reducer takes action ##
    #######################################
    
    

    avg_loss, avg_acc = avg_loss/n, avg_acc/n
    
    #########################################################
    # Save the best model that has the highest val accuracy #
    #########################################################
    if val_acc.item() > best_val_acc.item():
      print(f"\nPrev Best Val Acc: {best_val_acc} < Cur Val Acc: {val_acc}")
      print("Saving the new best model...")
      
      #save the model with torch.save
      torch.save({})
    
      best_val_acc = val_acc
      print("Finished saving model\n")
        
    # Print the metrics (should be same on all machines)
    print(f"\n(Epoch {epoch+1}/{epochs}) Time: {total_time}s")
    print(f"(Epoch {epoch+1}/{epochs}) Average train loss: {avg_loss}, Average train accuracy: {avg_acc}")
    print(f"(Epoch {epoch+1}/{epochs}) Val loss: {val_loss}, Val accuracy: {val_acc}")  
    print(f"(Epoch {epoch+1}/{epochs}) Current best val acc: {best_val_acc}\n")  

if __name__ == "__main__":
    torch.hub.set_dir('/tmp') # remove when not running here 
    hp = {"lr":1e-4, "batch_size":16, "epochs":5}
    train_set, val_set = load_datasets()
    train_dataloader, val_dataloader, test_dataloader = construct_dataloaders(train_set, val_set, hp["batch_size"], True)
    resnet = getResNet()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device

    resnet.to(device)
    
    # For saving the trained model
    model_folder_path = os.getcwd()+"/output_model/"
    os.makedirs(model_folder_path,exist_ok=True)

    # filename for our best model
    checkpoint_file = model_folder_path+"best_model.pt"

    # load the checkpoint that has the best performance in previous experiments
    prev_best_val_acc = None
    checkpoint_file = model_folder_path+"best_model.pt"
    if os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file, device)
        prev_best_val_acc = checkpoint['accuracy']
 
