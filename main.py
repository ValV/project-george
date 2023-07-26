import clip
import faiss
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class DatasetSingle(Dataset):
    def __init__(self, path_image, transform=None):
        self.path_image = path_image
        self.transform = transform
        self.max_size = 512

    def __getitem__(self, i):
        image = Image.open(self.path_image)
        label = torch.tensor([0])
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)
        return image, label

    def __len__(self):
        return 1


def predict(path_model, path_index_positive, path_index_negative, path_images, verbose=False):
    if isinstance(path_images, (dict, list, tuple, set)):
        path_images = tuple(path_images)
    else:
        path_images = (path_images, )
    device = torch.device('cpu')

    model_clip, preprocess = clip.load('ViT-B/32', device)
    model_metric = torch.jit.load(path_model)

    index_positive_total = faiss.read_index(path_index_positive)
    index_negative_total = faiss.read_index(path_index_negative)

    with torch.no_grad():
        model_clip.eval()
        for path_image in path_images:
            for images, labels in DataLoader(DatasetSingle(path_image,
                                                           preprocess),
                                            batch_size=1):
                if (verbose):
                    print(f"Extracting CLIP embeddings...")
                features = model_clip.encode_image(images.to(device))
                if (verbose):
                    print(f"Compressing embeddings...")
                features_latent = model_metric(features)
                if (verbose):
                    print(f"Matching against positive index...")
                distance_p, index_p = index_positive_total.search(
                    features_latent, 1
                )
                if (verbose):
                    print(f"Matching against negative index...")
                distance_n, index_n = index_negative_total.search(
                    features_latent, 1
                )
                if (verbose):
                    print(f"Compare results:")
                result_true = distance_p > distance_n
                result = result_true.squeeze().item()
                print(f"'{path_image}' {['does NOT look', 'LOOKS'][result]} like St. George")
def main(args):
    predict(args.path_model, args.path_index_positive, args.path_index_negative,
            args.path_images, args.verbose)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='path_model', required=True,
                        metavar='path', help='path to metric model')
    parser.add_argument('-p', '--positive', dest='path_index_positive',
                        required=True, metavar='path',
                        help='path to positive index')
    parser.add_argument('-n', '--negative', dest='path_index_negative',
                        required=True, metavar='path',
                        help='path to negative index')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbosely display progress')
    parser.add_argument('path_images', nargs=1, metavar='path',
                        help='path to the image of George/Not-George')
    main(parser.parse_args())
