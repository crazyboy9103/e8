import time
import copy
import torch 

def train_model(model, device, dataloader, criterion, optimizer, epoch):
    since = time.time()

    best_acc = 0.0

    
    print('Epoch {}'.format(epoch))
    print('-' * 10)

    # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
    model.train()
    

    running_loss = 0.0
    running_corrects = 0

    # 데이터를 반복
    for inputs, labels in dataloader:

        inputs = list(image.to(device) for image in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(inputs, targets)
        model.train()

        losses = sum(loss for loss in loss_dict.values())
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 매개변수 경사도를 0으로 설정
        optimizer.zero_grad()

        # 순전파
        # 학습 시에만 연산 기록을 추적
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 학습 단계인 경우 역전파 + 최적화
            loss.backward()
            optimizer.step()

        # 통계
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / len(dataloader)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))

    # 모델을 깊은 복사(deep copy)함

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    #model.load_state_dict(best_model_wts)
    return model     