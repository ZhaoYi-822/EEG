import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import model



FEATURE_CSV = "EEG_features_normalized.csv"
LABEL_CSV = "labels.csv"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data=pd.read_csv('EEG_dataset/cdl_eeg_dataset.csv')
test_data=pd.read_csv('EEG_dataset/cdl_eeg_dataset.csv')


x_train=train_data.iloc[:,:98]
y_train=train_data.iloc[:,98]

x_test=test_data.iloc[:,:98]
y_test=test_data.iloc[:,98]

x_train= x_train.reshape((-1, 98, 1))
x_test_seq = x_test.reshape((-1, 98, 1))





def train_torch_model(model, x_train, y_train, x_test, y_test, epochs, name):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")


    model.eval()
    preds_all, true_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds = model(xb)
            preds_all.extend(preds.argmax(1).cpu().numpy())
            true_all.extend(yb.numpy())

    acc = accuracy_score(true_all, preds_all)
    print(classification_report(true_all, preds_all))
    torch.save(model.state_dict(), "models/"+name+".pth")
    return acc




