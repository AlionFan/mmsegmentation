import argparse
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


class ONNXWrapper(torch.nn.Module):
    def __init__(self, segmentor):
        super().__init__()
        self.segmentor = segmentor

    def forward(self, x):
        batch_img_metas = []
        h, w = x.shape[2:]
        for _ in range(x.shape[0]):
            batch_img_metas.append(
                dict(
                    img_shape=(h, w, 3),
                    ori_shape=(h, w, 3),
                    pad_shape=(h, w, 3),
                    scale_factor=(1.0, 1.0),
                    flip=False))
        return self.segmentor.encode_decode(x, batch_img_metas)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out', default='model.onnx')
    parser.add_argument('--input-shape', default='1,3,512,512')
    parser.add_argument('--opset', type=int, default=11)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    segmentor = MODELS.build(cfg.model)
    load_checkpoint(segmentor, args.checkpoint, map_location='cpu')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    segmentor.to(device).eval()
    wrapper = ONNXWrapper(segmentor)

    shape = tuple(map(int, args.input_shape.split(',')))
    dummy = torch.randn(shape, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=['input'],
        output_names=['logits'],
        opset_version=args.opset,
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        do_constant_folding=True)

if __name__ == '__main__':
    main()
