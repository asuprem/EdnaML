import torch

class VAEGANOptimizerBuilder:
  def __init__(self,base_lr, lr_bias, gpus, weight_decay, weight_bias):
    self.base_lr = base_lr
    self.gpus = gpus
    #self.weight_decay = weight_decay
    #self.lr_bias = lr_bias
    #self.weight_bias = weight_bias


  def build(self, vaegan_model, _name = 'Adam', **kwargs):
    
    optimizer = __import__('torch.optim', fromlist=['optim'])
    optimizer = getattr(optimizer, _name)
    
    encoder_opt = optimizer(vaegan_model.Encoder.parameters(), lr = self.base_lr, **kwargs)
    decoder_opt = optimizer(vaegan_model.Decoder.parameters(), lr = self.base_lr, **kwargs)
    discriminator_opt = optimizer(vaegan_model.Discriminator.parameters(), lr = self.base_lr, **kwargs)
    autoencoder_opt = optimizer(list(vaegan_model.Encoder.parameters()) + list(vaegan_model.Decoder.parameters()), lr = self.base_lr, **kwargs)
    latent_opt = optimizer(vaegan_model.LatentDiscriminator.parameters(), lr = self.base_lr, **kwargs)

    return {"Encoder":              encoder_opt, \
            "Decoder":              decoder_opt, \
            "Discriminator":        discriminator_opt, \
            "Autoencoder":          autoencoder_opt, \
            "LatentDiscriminator":  latent_opt}


    """
    params = []
    for key, value in model.named_parameters():
      if value.requires_grad:
        if "bias" in key:
            learning_rate = self.base_lr * self.lr_bias
            weight_decay = self.weight_decay * self.weight_bias
        else:
            learning_rate = self.base_lr * self.gpus
            weight_decay = self.weight_decay
        params += [{"params": [value], "lr":learning_rate, "weight_decay": weight_decay}]
    optimizer = __import__('torch.optim', fromlist=['optim'])
    optimizer = getattr(optimizer, _name)
    optimizer = optimizer(params, **kwargs)
    return optimizer  
    """