import torch
import torch.nn as nn
from torchvision import transforms
from model import PointNet,UNet_PointNet,UNet_input_PointNet,UNet_feature_PointNet
from utils import (
    RandRotation_z,
    RandomNoise,
    shuffle,
    ToTensor,
    get_loaders,
    pointnetloss,
)

# Hyperparameters etc.
bs=32
epochs = 250
lr = 0.001
save_every = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir='/home/meiyih/shape_recognition/ModelNet40_data/'

model = PointNet()#UNet_PointNet() #UNet_input_PointNet()#UNet_feature_PointNet()
model.to(device)
loss_fn = pointnetloss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
early_stopping = EarlyStopping(patience=20, verbose=True)
writer = SummaryWriter('runs/experiment_baseline')

train_transforms = transforms.Compose([
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])
test_transforms = transforms.Compose([
                    ToTensor()
                    ])


train_loader, test_loader = get_loaders(data_dir,32,train_transforms,test_transforms)

def train(train_loader, test_loader=None,  epochs=15):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
        
    
    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        av_tr_loss = 0
        train_acc = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            av_tr_loss += loss.item()
            
            # print statistics
            running_loss += loss.item()
            acc = torch.sum(torch.argmax(outputs.data,dim = 1) == labels)
            train_acc += acc
            if i % 100 == 9:    # print every 100 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f, accuracy: %.2f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss/ 100, 100*acc.item()/labels.size()[0]))
                    running_loss = 0.0

        model.eval()
        correct = total = 0
        av_test_loss = 0
        test_acc = 0
        # validation
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].type(torch.LongTensor).to(device)
                    outputs, m3x3, m64x64 = model(inputs.transpose(1,2))
                    loss_test = pointnetloss(outputs, labels, m3x3, m64x64)
                    av_test_loss += loss_test.item()
                    test_acc += torch.sum(torch.argmax(outputs.data,dim = 1) == labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Test accuracy: %d %%' % val_acc)

        writer.add_scalar('Test Loss',av_test_loss/(len(test_ds)//bs),epoch)
        writer.add_scalar('Test acc',test_acc.item()/len(test_ds),epoch)
        writer.add_scalar('Train Loss',av_tr_loss/(len(train_ds)//bs),epoch)
        writer.add_scalar('Train acc',train_acc.item()/len(train_ds),epoch)
        train_losses.append(av_tr_loss/(len(train_ds)//bs))
        test_losses.append(av_test_loss/(len(test_ds)//bs))
        train_accs.append(train_acc.item()/len(train_ds))
        test_accs.append(test_acc.item()/len(test_ds))
        print("TESTING")       
        print("Epoch [{}/{}] --> Testing Loss:{:.3f}, Testing Accuracy:{:.2f}\n".format(epoch+1,epochs,av_test_loss/(len(test_ds)//bs), 100*test_acc.item()/len(test_ds)))
        print("TRAINING")
        print("Epoch [{}/{}] --> Training Loss:{:.3f}, Training Accuracy:{:.2f}\n".format(epoch+1,epochs,av_tr_loss/(len(train_ds)//bs), 100*train_acc.item()/len(train_ds)))
        
        # save the model
        if (epoch+1)%save_every == 0:
            print("SAVING MODEL CHECKPOINT")
            torch.save(model.state_dict(), "baseline_ckpt/"+"save_baseline_"+str(epoch)+".pth")
        
        #end of one epoch
        mean_loss = (av_test_loss/(len(test_ds)//bs))
        early_stopping(mean_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            torch.save(model.state_dict(), 'baseline_early_stop_model.pth')
            break #early stopping applied

    writer.flush()
    return train_losses,test_losses,train_accs,test_accs 