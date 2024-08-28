from torch.utils.data import Dataset

class TextSimplificationDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        assert len(input_tensor) == len(target_tensor)
        self.input_data = input_tensor
        self.target_data = target_tensor

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]