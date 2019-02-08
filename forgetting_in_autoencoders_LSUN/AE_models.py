import torch
import torch.nn as nn

class AE1(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE1, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, code_size),
    )
    self.decoder = nn.Sequential(
      nn.Linear(code_size, 128)
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE2(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE2, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      nn.Linear(128, 128),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE3(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE3, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 192),
      nn.Linear(192, 128),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE4(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE4, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 128),
      linear_block(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      nn.Linear(128, 128),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE5(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE5, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 128),
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      nn.Linear(128, 128),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


class AE6(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE6, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      nn.Linear(code_size, 128),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE7(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE7, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 224),
      linear_block(224, 512),
      linear_block(512, 128),
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      nn.Linear(code_size, 128),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE8(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE8, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 224),
      linear_block(224, 512),
      linear_block(512, 128),
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      nn.Linear(128, 128),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE9(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE9, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 224),
      linear_block(224, 512),
      linear_block(512, 128),
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      linear_block(128, 512),
      linear_block(512, 256),
      nn.Linear(256, 128),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE10(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE10, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 128),
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      linear_block(128, 512),
      linear_block(512, 256),
      nn.Linear(256, 128),
    )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE11(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE11, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 512),
      linear_block(512, 1024),
      linear_block(1024, 512),
      linear_block(512, 128),
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      linear_block(128, 512),
      linear_block(512, 512),
      linear_block(512, 256),
      nn.Linear(256, 128),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE_LSUN(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE_LSUN, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(2048, 512),
      linear_block(512, 128),
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      linear_block(128, 512),
      nn.Linear(512, 2048),
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class AE12(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(AE12, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 128),
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      linear_block(128, 512),
      linear_block(512, 512),
      linear_block(512, 256),
      nn.Linear(256, 128),
    )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


