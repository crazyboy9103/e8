# +
from MRCNN import *

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = CustomDataset("/data/dataset/recsys/e8/data_1230", "dataset_1230.pt")  
    model = Model(num_classes=dataset.num_classes, device = device, model_name = "mrcnn_model_65.pt", batch_size=8, parallel=False) # if there is no ckpt to load, pass model_name=None 
    model.fit(dataset, max_epochs=10)

if __name__ == "__main__":
    main()
