from MRCNN import *

class MyModel(Model):
    def test(self, dataset):
        testloader = DataLoader(dataset = dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
        evaluate(self.model, 1, testloader, device=self.device)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
dataset = CustomDataset("/data/dataset/recsys/e8/data_1230", "dataset_1230.pt")
myModel = MyModel(num_classes=dataset.num_classes, device = device, model_name = "mrcnn_model_75.pt", batch_size=8, parallel=False) # if there is no ckpt to load, pass model_name=None 
myModel.test(dataset)