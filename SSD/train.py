# +
from SSD import *
def main():
    data_dir = "/data/dataset/recsys/e8/data_1230"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
    dataset = CustomDataset(data_dir, data_dir + "/ssd_data.pt")
    model = Model(num_classes=dataset.num_classes, device = device, model_name = "ssd_model_110.pt", batch_size=128, parallel=False) # if there is no ckpt to load, pass model_name=None 
    model.fit(dataset, max_epochs=20)

if __name__ =="__main__":
    main()
