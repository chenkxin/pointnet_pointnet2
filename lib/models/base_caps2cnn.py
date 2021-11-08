import torch

from .block import *  # TODO: dirty import
from torch.autograd import Variable

class ModelSphericalCaps(BaseModel):
    def __init__(
        self,nclasses, b_in, primary, hidden, d_in=6, use_residual_block=True, **kwargs
    ):
        """
        S^2 data --> Initial Residual Layer --> Primary Capsule Layer -->
        Capsule Conv Layers --> SO(3) integrate --> L2 norm --> output
        Args:
            b_in: input spherical bandwidth
            primary: list of tuple, which produce primary capsules by conv
            hidden: list(tuple(int*3)), [(n_out_caps, d_out_caps, b_out), ...]
            d_in: # of channels of input
            use_residual_block: bool
            **kwargs:

        Example:
            1. directly instantiation from the model
            model = ModelSphericalCaps(
                        b_in=32,
                        primary=[
                            (50,32), # (d_out, b_out) S^2 conv block or residual block
                            (100,25), # (d_out, b_out) SO(3) conv block or residual block
                            (5,10,22) # (n_out_caps, d_out_caps, b_out)
                        ],
                        hidden = [
                            (5,10,22),
                            (5,10,7),
                            (5,10,7),
                            (nclasses,16,7)
                        ],
                        d_in = 6,
                    )
            2. inherited from this model, please refer to model_caps.py or smnist.py

        """
        super().__init__()
        assert len(hidden) > 0, "must have hidden layer!"
        assert (
            len(primary[-1]) == 3
        ), "primary capsule info is wrong, except:[...,(n_out_caps, d_out_caps, b_out)], but get: [...,{}]".format(
            primary[-1]
        )
        assert len(primary) >= 2, "must have two feature extractor now"
        self.b_in = b_in
        self.d_in = d_in
        self.use_residual_block = use_residual_block
        self.primary = primary
        self.nclasses = nclasses
        # Primary layers
        self.primary_layers = self._init_primary_layers(primary)

        # Capsule Conv Layers + so(3) integrate
        self.capsule_conv_layers = self._init_caps_conv_layers(hidden)

        #Decoder network
        self.decoder=nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,d_in*2*b_in*2*b_in),
            #nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x, y=None):
        x = self.primary_layers(x)
        self.hidden = []
        for layer in self.capsule_conv_layers:
            x = layer(x)
            self.hidden.append(x)
        self.class_capsule = x
        x = torch.norm(x, dim=-1)  # vector length
        if y is None: # during testing, no label given,create one-hot coding using 'length'
            index = x.max(dim=1)[1]
            y = Variable(torch.zeros(x.size()).scatter_(1, index.view(-1,1).cpu().data,1.).cuda())
        else:
            y = torch.eye(self.nclasses).index_select(dim=0, index=y).cuda()
        #return F.log_softmax(x, dim=1)
        # smnist
        # reconstruction = 255 * self.decoder(torch.matmul(self.class_capsule.permute(0, 2, 1), y[:, :, None]).view(self.class_capsule.size(0), -1))
        # 3D model
        reconstruction = -3 * self.decoder(torch.matmul(self.class_capsule.permute(0,2,1),y[:,:,None]).view(self.class_capsule.size(0),-1))
        return x, y, reconstruction.view(-1,self.d_in*2*self.b_in*2*self.b_in), self.nclasses

    def _init_primary_layers(self, primary):
        primary_layers = []

        # Initial Residual Layer
        d1, b1 = primary[0]
        primary_layers.append(
            get_inital_block(self.d_in, d1, self.b_in, b1, self.use_residual_block)
        )
        # maybe hidden initial layer
        if len(primary) >= 3:
            in_layers = primary[0:-2]
            out_layers = primary[1:-1]
            for i, out_tuple in enumerate(out_layers):
                in_tuple = in_layers[i]
                d_in, b_in = in_tuple
                d_out, b_out = out_tuple
                primary_layers.append(
                    get_capsule_block(d_in, d_out, b_in, b_out, self.use_residual_block)
                )
        # Primary Capsule Layer
        n_pri_caps, d_pri_caps, b_pri_caps = primary[-1]
        primary_layers.append(
            PrimaryCapsuleLayer(
                in_features=primary[-2][0],
                num_out_capsules=n_pri_caps,
                capsule_dim=d_pri_caps,
                b_in=primary[-2][1],
                b_out=b_pri_caps,
                use_residual_block=self.use_residual_block,
            )
        )
        return nn.Sequential(*primary_layers)

    def _init_caps_conv_layers(self, hidden):
        """

        Args:
            hidden:  [(n_out_caps, d_out_caps, b_out), ...]
        Returns:

        """
        caps_conv_list = []
        total = [self.primary[-1], *hidden]
        in_layers = total[:-1]
        out_layers = total[1:]
        for i in range(len(in_layers)):
            in_layer, out_layer = in_layers[i], out_layers[i]
            n_in, d_in, b_in = in_layer
            n_out, d_out, b_out = out_layer
            caps_conv_list.append(
                ConvolutionalCapsuleLayer(
                    num_in_capsules=n_in,
                    in_capsule_dim=d_in,
                    num_out_capsules=n_out,
                    out_capsule_dim=d_out,
                    b_in=b_in,
                    b_out=b_out,
                    is_class=True if i == len(in_layers) - 1 else False,
                    use_residual_block=self.use_residual_block,
                )
            )
        return nn.ModuleList(caps_conv_list)

class CapsuleLoss(BaseModel):
    """Combine margin loss of capsule network."""
    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda

    def forward(self, logits, labels, x, x_recon, nclasses):
        # Shape of left / right / labels: (batch_size, num_classes)
        labels = torch.eye(nclasses).index_select(dim=0, index=labels).cuda()
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)
        margin_loss = margin_loss.mean()
        return margin_loss

class Capsule_recon(BaseModel):
    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5, lam_recon=0.0005):
        super(Capsule_recon, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.lam_recon = lam_recon

    def forward(self, logits, labels, x, x_recon, nclasses):
        # Shape of left / right / labels: (batch_size, num_classes)
        labels = torch.eye(nclasses).index_select(dim=0, index=labels).cuda()
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)
        margin_loss = margin_loss.mean()
        L_recon=nn.MSELoss()(x_recon, x.view(x.size(0),-1))
        return margin_loss + self.lam_recon * L_recon

