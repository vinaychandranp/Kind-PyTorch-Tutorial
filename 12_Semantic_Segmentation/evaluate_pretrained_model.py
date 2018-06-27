import os
import argparse
from PIL import Image

# Models
from FusionNet import FusionGenerator
from UNet import UnetGenerator

# Torch stuff
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils


parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, default=None, help="path to an image to evaluate")
parser.add_argument("--network", type=str, default="fusionnet", help="choose between fusionnet & unet")
args = parser.parse_args()


# Create model
if args.network == "fusionnet":
    generator = FusionGenerator(3, 3, 64)
elif args.network == "unet":
    generator = UnetGenerator(3, 3, 64)


# Load pre-trained model
# try:
checkpoint= torch.load('./model/{}.pkl'.format(args.network))
generator.load_state_dict(checkpoint['state_dict'])
#     print("\n--------model restored--------\n")
# except:
#     print("\n--------model not restored--------\n")
#     pass

# Load the image

def image_loader(image_name):
    loader = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

    image = Image.open(image_name)
    image = image.convert('RGB')
    image = loader(image).float().unsqueeze(0)
    return image


image = image_loader(args.image)

satellite_image, map_image = torch.chunk(image, chunks=2, dim=3)

x = Variable(satellite_image)
y_ = Variable(map_image)
y = generator.forward(x)


if not os.path.exists('output'):
    os.makedirs('output')

original_image_name = os.path.basename(args.image).split('.')[0]


v_utils.save_image(x.cpu().data, os.path.join('output', '{}_{}.png'.format(original_image_name,
                                                                              'satellite')))
v_utils.save_image(y_.cpu().data, os.path.join('output', '{}_{}.png'.format(original_image_name,
                                                                              'map')))
v_utils.save_image(y.cpu().data, os.path.join('output', '{}_{}_{}.png'.format(original_image_name,
                                                                              'generated',
                                                                              args.network)))
print('All done!')