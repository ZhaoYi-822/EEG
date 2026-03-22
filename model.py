
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.svm import SVC



def build_xgboost(n_classes):
    xg_model = XGBClassifier(
        booster='gbtree',
        n_estimators=230,
        max_depth=15,
        random_state=42,
        num_class=n_classes,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
    )
    return xg_model



def build_svm():

    svm_model=SVC(
        C=10,
        kernel='rbf',
        probability=True,
        decision_function_shape='ovo',
        random_state=42,
    )
    return svm_model




class LSTMModel(nn.Module):

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, n_classes: int = 25,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)




class CNNModel(nn.Module):
    def __init__(self, input_size: int, n_classes: int = 25,
                 dropout: float = 0.5):
        super().__init__()

        self.conv_block = nn.Sequential(

            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        conv_out_len = input_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * conv_out_len, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block(x))

