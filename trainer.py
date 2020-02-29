import torch
from tqdm import tqdm
from easydict import EasyDict


def train_model(train_config):
    def get_progress_bar(loader):
        return tqdm(enumerate(loader), total=len(loader))

    def optimizer_step(loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def to_device(images, labels):
        images = images.to(train_config.device)
        labels = labels.to(train_config.device)
        return images, labels

    def train_one_epoch(epoch):
        model.train()
        pbar = get_progress_bar(loaders.train)
        for i, (images, labels) in pbar:
            images, labels = to_device(images, labels)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            optimizer_step(loss)
            scheduler.step()

            pbar.set_description(f'train epoch {epoch} loss {loss.item():.4f}')

    def init_metric():
        return EasyDict(dict(
            total=0, correct=0
        ))
    
    def accumulate_metric(predicted, labels, metric):
        metric.total += labels.size(0)
        metric.correct += (predicted == labels).sum().item()
    
    def test_one_epoch(epoch):
        model.eval()
        pbar = get_progress_bar(loaders.test)
        metric = init_metric()           
        with torch.no_grad():
            for i, (images, labels) in pbar:
                images, labels = to_device(images, labels)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                accumulate_metric(predicted, labels, metric)
                
                pbar.set_description(f'test  epoch {epoch} accuracy {100*metric.correct/metric.total:.2f}%')
            save_model(accuracy=metric.correct/metric.total, epoch=epoch)

    def save_model(accuracy, epoch):
        nonlocal current_best
        data_to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_acc': accuracy,
        }
        torch.save(data_to_save, f'{train_config.model_path}.ckpt')
        if accuracy > current_best:
            print(f'val_acc improved from {current_best*100:.2f}% to {accuracy*100:.2f}')
            torch.save(data_to_save, f'{train_config.model_path}_best.ckpt')
            current_best = accuracy

    # training main code
    model = train_config.model
    optimizer = train_config.optimizer
    scheduler = train_config.scheduler
    criterion = train_config.criterion
    datasets = train_config.datasets
    
    # create data loaders
    loaders = EasyDict(dict(
        train=torch.utils.data.DataLoader(datasets.train, batch_size=train_config.batch_size, shuffle=True),
        test=torch.utils.data.DataLoader(datasets.test, batch_size=train_config.batch_size, shuffle=False),
    ))
    
    # start training
    current_best = float('-inf')
    for epoch in range(train_config.num_epochs):
        train_one_epoch(epoch)
        test_one_epoch(epoch)
