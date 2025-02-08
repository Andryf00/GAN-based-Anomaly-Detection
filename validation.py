
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.nn import Conv2d, ConvTranspose2d, Flatten, Linear, LeakyReLU, Dropout
import numpy as np
from sklearn.cluster import KMeans
import random
from skimage.transform import resize
from torchvision import datasets, transforms
import math 
import matplotlib.pyplot as plt
import torchvision.transforms.functional
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import FashionMNIST
from sklearn.metrics import roc_auc_score


class AutoEncoder(LightningModule):
    
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=22, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2d(in_channels=22, out_channels=6, kernel_size=4, stride=2, padding=1)
        self.conv3 = Conv2d(in_channels=6, out_channels=84, kernel_size=4, stride=2, padding=1)
        self.conv4 = Conv2d(in_channels=84, out_channels=24, kernel_size=4, stride=2, padding=1)
        self.up_conv1 = ConvTranspose2d(in_channels=24, out_channels=84, kernel_size=4, stride=2, padding=1)
        self.up_conv2 = ConvTranspose2d(in_channels=168, out_channels=6, kernel_size=4, stride=2, padding=1)
        self.up_conv3 = ConvTranspose2d(in_channels=12, out_channels=22, kernel_size=4, stride=2, padding=1)
        self.up_conv4 = ConvTranspose2d(in_channels=44, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.automatic_optimization=False
        self.l_relu = LeakyReLU()
        self.dropout = Dropout(0.5)

    def forward(self, x):
        #input size 32x32x1        
        c1 = self.dropout(self.l_relu(self.conv1(x)))
        c2 = self.dropout(self.l_relu(self.conv2(c1)))
        c3 = self.dropout(self.l_relu(self.conv3(c2)))
        c4 = self.dropout(self.l_relu(self.conv4(c3)))
        #c4 is the latent rapresentation
        u_c1 = self.l_relu(self.up_conv1(c4))
        #crop_conv_1 = get_cropping_shape(c4.shape[1], u_c1.shape[1])
        #c4_cropped = c4[:, crop_conv_1:-crop_conv_1, crop_conv_1:-crop_conv_1]
        u_c1 = torch.cat((u_c1,c3), 0)

        u_c2 = torch.relu(self.up_conv2(u_c1))
        #crop_conv_2 = get_cropping_shape(c3.shape[1], u_c2.shape[1])
        #c3_cropped = c3[:, crop_conv_2:-crop_conv_2, crop_conv_2:-crop_conv_2]
        u_c2 = torch.cat((u_c2,c2), 0)

        u_c3 = torch.relu(self.up_conv3(u_c2))
        #crop_conv_3 = get_cropping_shape(c2.shape[1], u_c3.shape[1])
        #c2_cropped = c2[:, crop_conv_3:-crop_conv_3, crop_conv_3:-crop_conv_3]
        u_c3 = torch.cat((u_c3,c1), 0)

        u_c4 = torch.relu(self.up_conv4(u_c3))
        #crop_conv_1 = get_cropping_shape(c1.shape[1], u_c4.shape[1])
        #c1_cropped = c1[:, crop_conv_1:-crop_conv_1, crop_conv_1:-crop_conv_1]
        #u_c4 = torch.cat((u_c4,c1_cropped), 1)   
        return u_c4
    
def kmeans(samples, size, total):
  k = 3
  for i in range(1):
    N = []
    A = []
    N_samp = []

    for j in range(total):
      if samples[j][1]==5: N.append(samples[j][0])
      else: 
        x = samples[j][0].squeeze(0).numpy().flatten()
        A.append(x)
    print(len(N),len(A))
    A1 = []
    A2 = []
    A3 = []
    
    N_samp = random.sample(N,1000)
    kmeans = KMeans(n_clusters=k).fit(A)
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] == 0: A1.append(np.reshape(np.array(A[i]), (32,32)))
        if kmeans.labels_[i] == 1: A2.append(np.reshape(np.array(A[i]), (32,32)))
        if kmeans.labels_[i] == 2: A3.append(np.reshape(np.array(A[i]), (32,32)))
    
    A1_sample_index = random.sample(range(1, len(A1)), int(size/6))
    A2_sample_index = random.sample(range(1, len(A2)), int(size/6))
    A3_sample_index = random.sample(range(1, len(A3)), int(size/6))

    A1_samp = [[A1[indx], 0] for indx in A1_sample_index]
    A2_samp = [[A2[indx], 0] for indx in A2_sample_index]
    A3_samp = [[A3[indx], 0] for indx in A3_sample_index]
    
    N_samp = [[np.reshape(np.array(img),(32,32)), 1] for img in N_samp]

    return N_samp + A1_samp + A2_samp + A3_samp

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [0.80, 0.20])
            print(self.mnist_train)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = FashionMNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = FashionMNIST(self.data_dir, train=False, transform=self.transform)
            print(len(self.mnist_predict))
            self.mnist_predict = kmeans(self.mnist_predict, 10000, 10000)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=1, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=1, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=1, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=1, shuffle=False)



class Discriminator(LightningModule):

    def __init__(self):
        super().__init__()
        """self.conv1 = Conv2d(in_channels=1, out_channels=22, kernel_size=4, stride=1, padding=0)
        self.conv2 = Conv2d(in_channels=22, out_channels=6, kernel_size=4, stride=2, padding=0)
        self.conv3 = Conv2d(in_channels=6, out_channels=84, kernel_size=4, stride=1, padding=0)
        self.conv4 = Conv2d(in_channels=84, out_channels=24, kernel_size=4, stride=1, padding=0,padding_mode='reflect')
        self.conv5 = Conv2d(in_channels=24, out_channels=84, kernel_size=4, stride=1, padding=0,padding_mode='reflect')
        #self.conv6 = Conv2d(in_channels=84, out_channels=1, kernel_size=4, stride=1, padding=0, padding_mode='reflect')"""
        self.conv1 = Conv2d(in_channels=1, out_channels=22, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2d(in_channels=22, out_channels=6, kernel_size=4, stride=1, padding=0)
        self.conv3 = Conv2d(in_channels=6, out_channels=84, kernel_size=4, stride=1, padding=0)
        self.conv4 = Conv2d(in_channels=84, out_channels=24, kernel_size=4, stride=1, padding=0)
        self.conv5 = Conv2d(in_channels=24, out_channels=84, kernel_size=4, stride=1, padding=0)
        self.conv6 = Conv2d(in_channels=84, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.flatten = Flatten()
        self.fc = Linear(16, 1)
        self.automatic_optimization=False
        self.l_relu = LeakyReLU()
        self.dropout = Dropout(0.5)


    def forward(self, x):
        #input size 32x32x1        
        c1 = self.l_relu(self.conv1(x))
        c2 = self.l_relu(self.conv2(c1))
        c3 = self.l_relu(self.conv3(c2))
        c4 = self.l_relu(self.conv4(c3))
        x = self.l_relu(self.conv5(c4))
        x = self.l_relu(self.conv6(x))
        blah = torch.sigmoid(x)
        #blah = torch.mean(blah, dim=0)
        pred = torch.sigmoid(self.fc(self.flatten(x)))
        return blah, pred

model=Discriminator()
x=model(torch.ones((1,32,32)))

"""**GAN**

"""

class Gan(LightningModule):
    def __init__(self):
      super().__init__()
      self.save_hyperparameters()
      self.automatic_optimization=False

      # networks
      self.autoencoder = AutoEncoder()
      #self.autoencoder.train()
      self.normal_disc = Discriminator()
      #self.normal_disc.train()
      self.anomaly_disc = Discriminator()
      #self.anomaly_disc.train()
      self.l1_loss = torch.nn.L1Loss()
      self.ones = torch.ones(size=(1,4,4), device=self.device)
      self.zeros = torch.zeros(size=(1,4,4), device=self.device)

    def forward(self, z):
        return self.autoencoder(z)

    def training_step(self, sample, batch_idx):
        x, normal = sample
        opt_autoencoder, opt_anomaly, opt_normal = self.optimizers()
        sch = self.lr_schedulers()
        sch.step()
        x_reconstructed = self.forward(x) 
        
        label_1 = 1
        label_0 = 0
        if batch_idx%6000<10:      
          print("label", normal, "recon_error", self.l1_loss(x, x_reconstructed))
          fig, (ax1, ax2) = plt.subplots(1, 2)
          ax1.imshow(x.squeeze(0).detach().numpy())
          ax2.imshow(x_reconstructed.squeeze(0).detach().numpy())

          # Add a description
          ax1.set_xlabel("Original"+str(normal))
          plt.show()

        x_reconstructed = self.forward(x) 



        if normal:  
            patches_n, y_reconstructed_n = self.normal_disc(x_reconstructed.detach())
            patches_orig_n, y_original_n = self.normal_disc(x)
            #discriminator loss

            disc_loss_n = (y_original_n-label_1)**2 + (y_reconstructed_n-label_0)**2
            #print("NORMAL DISC LOSS: ", disc_loss)
            opt_normal.zero_grad()
            self.manual_backward(disc_loss_n)#, retain_graph=True)
            #print(self.normal_disc.conv5.weight)
            opt_normal.step()

        elif not normal: 
            patches_a, y_reconstructed_a = self.anomaly_disc(x_reconstructed.detach())
            patches_orig_a, y_original_a = self.anomaly_disc(x)
            #discriminator loss
            disc_loss_a = (y_original_a-label_1)**2 + (y_reconstructed_a-label_0)**2
            #print("ANOMALY DISC LOSS: ", disc_loss)
            opt_anomaly.zero_grad()
            self.manual_backward(disc_loss_a)#, retain_graph=True)
            opt_anomaly.step()

        #Generator losses
        if normal: 
            patches_n, y_n = self.normal_disc(x_reconstructed)
            loss_adv_n = torch.mean((patches_n-self.ones)**2)
            #print("LOSS_ADV_NORMAL:", loss_adv_normal)
            #latent vector loss
                  
            loss_enc = self.l1_loss(x_reconstructed, self.autoencoder(x_reconstructed.detach().clone()))
            loss_recon_n = self.l1_loss(x, x_reconstructed)
            
            #loss_patch
            patch_losses = torch.zeros(9)
            k = 0

            #for i in range(32,16,-8):    #range is 24 because with patch size 16 we overshoot 
            for i in range(0,24,8):
                for j in range(0,24,8):
                    """plt.imshow(x_reconstructed.squeeze(0).detach().numpy()[8:24,0:16])
                    plt.show()
                    plt.imshow(x_reconstructed.squeeze(0).detach().numpy()[8:24,8:24])
                    plt.show()"""
                    
                    cropped_x = torchvision.transforms.functional.crop(x, i, j, 16, 16)
                    cropped_x_recon = torchvision.transforms.functional.crop(x_reconstructed, i, j, 16, 16)
                  
                    """plt.imshow(cropped_x_recon.squeeze(0).detach().numpy())
                    plt.show()"""
                    loss_patch = self.l1_loss(cropped_x, cropped_x_recon)
                    #print("LOSS_PATCH", i, j, loss_patch)
                    patch_losses[k] = loss_patch.clone()

                    k+=1

            values, idxs = torch.topk(patch_losses, 3)
            patch_loss = torch.mean(values)
            generator_loss_n = 1.5*loss_recon_n + 0.5*loss_adv_n + 1.5*patch_loss + 0.5*loss_enc
            loss_abc_a = 0
            #print("NORMAL GEN LOSS: ", generator_loss)           
            opt_autoencoder.zero_grad()
            self.manual_backward(generator_loss_n)#,retain_graph=True)
            opt_autoencoder.step() 
            
        elif not normal: 
            patches_a, y_a = self.anomaly_disc(x_reconstructed)
            loss_recon_a = self.l1_loss(x,x_reconstructed)
            loss_adv_a = torch.sum((patches_a-self.zeros)**2)
            
            #ABC loss
            loss_abc_a = -math.log(1 - math.e**(-loss_recon_a))

            generator_loss_a = 0.5*loss_abc_a + 1*loss_adv_a
            loss_patch = loss_enc = 0
            #print("ANOMALY GEN LOSS: ", generator_loss)
        
            opt_autoencoder.zero_grad()
            self.manual_backward(generator_loss_a)#,retain_graph=True)
            opt_autoencoder.step() 

        if normal: return {"label":normal,"disc_loss": disc_loss_n,"generator_loss": generator_loss_n,"loss_recon": loss_recon_n,"loss_adv": loss_adv_n,"loss_patch": loss_patch,"loss_enc": loss_enc,"loss_abc": loss_abc_a}
        else: return {"label":normal,"disc_loss": disc_loss_a,"generator_loss": generator_loss_a,"loss_recon": loss_recon_a,"loss_adv": loss_adv_a,"loss_patch": loss_patch,"loss_enc": loss_enc,"loss_abc": loss_abc_a}
         #scheduler.step()
        
    def training_epoch_end(self, outputs) -> None:
        n_n= n_a = disc_loss_n = generator_loss_n = loss_recon_n = loss_adv_n = loss_patch = loss_enc = disc_loss_a = generator_loss_a = loss_recon_a = loss_abc = 0
        for output in outputs:
            if output['label'] == 1:
                n_n+=1
                disc_loss_n += output["disc_loss"]
                generator_loss_n += output["generator_loss"]
                loss_recon_n += output["loss_recon"]
            else:
                n_a+=1
                disc_loss_a += output["disc_loss"]
                generator_loss_a += output["generator_loss"]
                loss_recon_a += output["loss_recon"]
                loss_abc += output["loss_abc"]
        logs={"normal discriminator loss": disc_loss_n/n_n,"anomaly discriminator loss": disc_loss_a/n_a, "normal reconstruction loss": loss_recon_n/n_n, \
                  "anomaly reconstruction loss": loss_recon_a/n_a, "anomaly loss abc": loss_abc/n_a, "normal generator loss": generator_loss_n/n_n, "anomaly generator loss": generator_loss_a/n_a}
        print(logs)
        wandb.log(logs)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y!=1: y = 0
        #if not y: x+=torch.randn(x.size()) * 0.3
        x_reconstructed = self.forward(x)
        #should it be like this? or for validation we only use anomaly_disc?
        patches, y_hat = self.normal_disc(x_reconstructed)
        #print("label", y, "recon_error", self.l1_loss(x, x_reconstructed), "predicted", y_hat)
        """if batch_idx%500<10: 
            print(batch_idx)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(x.squeeze(0).detach().numpy())
            ax2.imshow(x_reconstructed.squeeze(0).detach().numpy())

            # Add a description
            ax1.set_xlabel("Original"+str(y))
            ax2.set_xlabel("Reconstructed:"+str(y_hat))
            plt.show()"""
        return {"label": y, "recon_error": self.l1_loss(x, x_reconstructed), "predicted": y_hat}

    def validation_epoch_end(self, validation_step_outputs):
        if(len(validation_step_outputs)>5):
          all_preds = validation_step_outputs
          anomaly_error = []
          normal_error = []
          n_zeros = 0
          n_anomalies = 0
          n_ones = 0
          n_normals = 0
          for x in validation_step_outputs:
              if x['label']==0:
                  if x['predicted']<0.5: n_zeros += 1
                  n_anomalies += 1
                  anomaly_error.append(x['recon_error'])
              if x['label']==1:
                  if x['predicted']>=0.5: n_ones += 1
                  n_normals += 1
                  normal_error.append(x['recon_error'])
              

          print("____"+str(self.current_epoch)+"____")
          #print("ANOM ERRORS: ", anomaly_error)
          print("anomaly avg error:", sum(anomaly_error)/len(anomaly_error), "accuracy:", n_zeros, "/", n_anomalies)
          print("____")
          print("normal avg error:", sum(normal_error)/len(normal_error), "accuracy:", n_ones, "/", n_normals)

    def configure_optimizers(self):
        opt_autoencoder = torch.optim.Adam(self.autoencoder.parameters(), lr=0.0000625)#lr=0.0001 )
        opt_anomaly = torch.optim.Adam(self.anomaly_disc.parameters(), lr=0.0001)#, momentum=0.001)
        opt_normal = torch.optim.Adam(self.normal_disc.parameters(), lr=0.0001)#, momentum=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_autoencoder, [50], gamma=0.5)

        return [opt_autoencoder, opt_anomaly, opt_normal], [scheduler]



if __name__=="main":
    print("QUAAAAAAA")
net = Gan()
net = net.load_from_checkpoint("epoch=15-step=192000-v2.ckpt")

print("laoded net")
dm = MNISTDataModule()
dm.setup("predict")
anomaly_error = []
normal_error = []
y_true = []
y_score = []
n_zeros = 0
n_anomalies = 0
n_ones = 0
n_normals = 0

for i, batch in enumerate(dm.predict_dataloader()):

    x, label = batch
    x_reconstructed = net.autoencoder(x)
    diff = x - x_reconstructed
    """if label==1:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(x.squeeze(0).detach().numpy())
        ax2.imshow(diff.squeeze(0).detach().numpy())

        # Add a description
        ax1.set_xlabel("Original")
        ax2.set_xlabel("Reconstructed:")
        plt.show()"""
    l1_loss = torch.nn.L1Loss()
    error=l1_loss(x,x_reconstructed)
    y_true.append(label.detach().numpy())
    if label==0:
        if error>0.150: 
            n_zeros += 1
            y_hat = 0
        else: y_hat=1
        n_anomalies += 1
        anomaly_error.append(error.detach().numpy())
    elif label==1:
        if error<=0.150: 
            n_ones += 1
            y_hat = 1
        else: y_hat=0
        n_normals += 1
        normal_error.append(error.detach().numpy())
    y_score.append(y_hat)


print("____")
#print("ANOM ERRORS: ", anomaly_error)
print("anomaly avg error:", sum(anomaly_error)/len(anomaly_error), "accuracy:", n_zeros, "/", n_anomalies)
print("____")
print("normal avg error:", sum(normal_error)/len(normal_error), "accuracy:", n_ones, "/", n_normals)

print(roc_auc_score(y_true, y_score))

# Set the range and step of the x and y axes
x_min, x_max, x_step = 0, 15, 0.05
y_min, y_max, y_step = 0, 1000, 10

# Create the histogram plots
plt.hist(anomaly_error, bins=np.arange(x_min, x_max + x_step, x_step), alpha=0.5, color='blue')
plt.hist(normal_error, bins=np.arange(x_min, x_max + x_step, x_step), alpha=0.5, color='red')

# Customize the x and y axes
plt.xticks(np.arange(x_min, x_max + x_step, x_step))
plt.yticks(np.arange(y_min, y_max + y_step, y_step))

# Add labels and a legend
plt.xlabel('Values')
plt.ylabel('Counts')
plt.legend(['anomaly_error', 'normal_error'])

# Show the plot
plt.show()

from sklearn.linear_model import LinearRegression
x = np.array(anomaly_error+normal_error).reshape(-1, 1)
# y = 1 * x_0 + 2 * x_1 + 3
y = np.array([0]*len(anomaly_error) + [1]*len(normal_error))
reg = LinearRegression().fit(x, y)
print(reg.score(x, y))
print(reg.coef_)
pred=reg.predict(x)
print(pred)
print(roc_auc_score(pred, y))
