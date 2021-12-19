import gdown

url = 'https://drive.google.com/u/2/uc?id=1djoCq7N8_B6733nBrNROHqUpLcsJKPrO&export=download'
output = 'data/potenit/3D_Localization.zip'
gdown.download(url, output, quiet=False)