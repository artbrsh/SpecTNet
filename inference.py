"""Run SpecTNet inference on an image using an explicitly supplied checkpoint."""

import argparse

import torch
from PIL import Image

from config import get_config
from data import build_transform
from models import build_model, prepare_for_inference
from utils import add_common_args


def main():
    parser = argparse.ArgumentParser(description='SpecTNet image classification')
    add_common_args(parser)
    parser.add_argument('--image', required=True, help='path to the input image')
    args = parser.parse_args()
    if not args.resume:
        parser.error('--resume is required and must point to a model checkpoint')

    config = get_config(args)
    model = build_model(config)
    # Use trusted training checkpoints, which may also contain configuration objects.
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    # Inference requires all model parameters, including the classifier, to match.
    model.load_state_dict(checkpoint['model'], strict=True)
    device = torch.device(
        'cpu' if args.only_cpu or not torch.cuda.is_available() else 'cuda')
    model = prepare_for_inference(model.to(device), config)

    transform = build_transform(is_train=False, config=config)
    with Image.open(args.image) as image:
        batch = transform(image.convert('RGB')).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=-1)
        scores, indices = probabilities.topk(min(5, probabilities.shape[-1]), dim=-1)

    print(args.image)
    # Class IDs follow the training dataset ordering, including custom datasets.
    for score, index in zip(scores[0].cpu().tolist(), indices[0].cpu().tolist()):
        print(f'class {index}: {score:.6f}')


if __name__ == '__main__':
    main()
