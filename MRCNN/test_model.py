from MRCNN import *
import argparse

class MyModel(Model):
    def test(self, dataset):
        testloader = DataLoader(dataset = dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
        evaluate(self.model, 1, testloader, device=self.device)


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="test model")
    parser.add_argument("--dir", default="/dataset", type=str, help="path to the dataset folder")
    parser.add_argument("--data", default="mrcnn_data.pt", type=str, help="dataset name to create OR load")
    parser.add_argument("--model", default="mrcnn_model_10.pt", type=str, help="model name to test")
    parser.add_argument("--batch", default=8, type=int, help="the batch size")

    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
    dataset = CustomDataset(args.dir, args.data)
    myModel = MyModel(num_classes=dataset.num_classes, device = device, model_name = args.model, batch_size=args.batch, parallel=False)
    myModel.test(dataset)
