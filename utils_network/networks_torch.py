import torch.nn as nn

class Convolutional_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # padding=1 means strides=1, therefore the output has the same size as the input.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride=1, bias=False)
        
        self.convblock = nn.Sequential(self.conv1, 
                                       self.batchnorm, 
                                       self.activ, 
                                       self.conv2, 
                                       self.batchnorm, 
                                       self.activ)

    def forward(self, x):
        return self.convblock(x)

class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.convblock = Convolutional_Layer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.1)

        self.downsampl = nn.Sequential(self.convblock, self.pool, self.drop)

    def forward(self, x):
        x2 = self.convblock(x)
        x = self.pool(x2)
        x = self.drop(x)
        return x, x2
        
class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        #self.convt = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.convt = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride=1))
        
        self.drop = nn.Dropout2d(p=0.1)
        self.convblock = Convolutional_Layer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.convt(x1)
        x = self.concat = torch.cat([x1, x2], dim=1)        
        x = self.drop(x)
        x = self.convblock(x)

        return x
    
class UNet(nn.Module):
    def __init__(self, img_shape, params):
        super(UNet, self).__init__()
        self.img_shape = img_shape
        self.params = params
        
        self.nr_layers = params['depth']
        self.latent_dim = params['coarse_dim']
        self.final_activation = params['final_activation']

        # TODO: add an assertion that check if the depth is compatible with the image size
        assert self.img_shape[-1] % 2**(self.nr_layers-1) == 0
        
        # collect the layers for a better visualisation
        self.layers = nn.ModuleDict() 
        
        # --- Encoder: downsampling --- 
        for i_l in range(self.nr_layers):
            if(i_l == 0):
                in_channels = 1
                out_channels = self.latent_dim//2**(self.nr_layers)
            else:
                in_channels = self.latent_dim//2**(self.nr_layers-i_l+1)
                out_channels = self.latent_dim//2**(self.nr_layers-i_l)
            
            self.layers['E%d' %i_l] = Downsampling(in_channels=in_channels, out_channels=out_channels, kernel_size=self.params['kernel_size'])
            
        # --- Bottom layer: latent space --- 
        self.layers['B'] = Convolutional_Layer(in_channels=out_channels, out_channels=self.latent_dim, kernel_size=self.params['kernel_size'])
        
        # --- Decoder: upsampling --- 
        for i_l in range(self.nr_layers)[::-1]:
            in_channels = self.latent_dim//2**(self.nr_layers-i_l-1)
            out_channels = self.latent_dim//2**(self.nr_layers-i_l)
            
            #print(i_l, ':', in_channels, out_channels)
            self.layers['D%d' %i_l] = Upsampling(in_channels=in_channels, out_channels=out_channels, kernel_size=self.params['kernel_size'])
        
        # --- Output layer ---
        if(self.final_activation):
            self.layers['CL'] = Convolutional_Layer(in_channels=out_channels, out_channels=1, kernel_size=self.params['kernel_size'])
            self.layers['O'] = nn.Sigmoid()
        else:
            self.layers['O'] = Convolutional_Layer(in_channels=out_channels, out_channels=1, kernel_size=self.params['kernel_size'])
        
    def forward(self, x):

        decoder_input = {} 
        
        for i_l, layer_name in enumerate(self.layers.keys()):
            if ('E' in layer_name):
                x, x2 = self.layers[layer_name](x)
                decoder_input[layer_name.replace('E', 'D')] = x2
            elif('D' in layer_name):
                x2 = decoder_input[layer_name]
                x = self.layers[layer_name](x, x2)
            else:
                x = self.layers[layer_name](x)
            #print(layer_name, x.detach().numpy().shape)
            
        return x