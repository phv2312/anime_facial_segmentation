import glob
import os

import cv2
import torch.types
from mit_semseg.models import ModelBuilder
from mit_semseg.inference.io_utils import parse_yml
from mit_semseg.inference.image_utils import *


class AnimeInference:
    def __init__(self, encoder_path, decoder_path, config_path):
        assert os.path.exists(encoder_path), 'encoder path: %s should be existed!!!' % encoder_path
        assert os.path.exists(decoder_path), 'decoder path: %s should be existed!!!' % decoder_path
        assert os.path.exists(config_path), 'config path: %s should be existed!!!' % config_path

        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.config_path  = config_path

        self.config = parse_yml(config_path)
        # over-ride
        self.config['MODEL']['weights_encoder'] = encoder_path
        self.config['MODEL']['weights_decoder'] = decoder_path

        #
        # load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder, self.decoder = self.__load_model(self.config)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        #
        # load input image's configuration
        self.img_sizes = eval(self.config['DATASET']['imgSizes'])
        self.n_cls  = self.config['DATASET']['num_class']


    def __load_model(self, config):
        net_encoder = ModelBuilder.build_encoder(
            arch=config['MODEL']['arch_encoder'],
            fc_dim=config['MODEL']['fc_dim'],
            weights=config['MODEL']['weights_encoder'])
        net_decoder = ModelBuilder.build_decoder(
            arch=config['MODEL']['arch_decoder'],
            fc_dim=config['MODEL']['fc_dim'],
            num_class=config['DATASET']['num_class'],
            weights=config['MODEL']['weights_decoder'],
            use_softmax=True)
        net_encoder.eval()
        net_decoder.eval()
        return net_encoder, net_decoder

    def process(self, image_path):
        #
        # Image processing
        img = Image.open(image_path).convert('RGB')
        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.img_sizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        1000. / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = round2nearest_multiple(target_width, 32)
            target_height = round2nearest_multiple(target_height, 32)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        img_ori = np.array(img)
        img_multi_scale = [x.contiguous() for x in img_resized_list]
        n_scale = len(img_multi_scale)

        #
        # Model running
        with torch.no_grad():
            scores = torch.zeros(1, self.n_cls, ori_height, ori_width)
            scores = scores.to(self.device)

            for img in img_multi_scale:
                img = img.to(self.device)
                output_encoder = self.encoder(img, return_feature_maps=True)
                pred_tmp = self.decoder(output_encoder, segSize=(ori_height, ori_width), use_log_softmax=False)
                pred_tmp = torch.sigmoid(pred_tmp)

                scores = scores + pred_tmp / n_scale

        # Formatting output
        mask_cls_dict = {}
        for cls_id in range(self.n_cls):
            mask_cls = scores[0][cls_id].cpu().numpy()
            mask_cls_dict[cls_id] = mask_cls

        return [mask_cls_dict]

def single_test():
    encoder_path = r"C:\Users\Cinnamon\Desktop\geek_resnet50_upernet\encoder_epoch_5.pth"
    decoder_path = r"C:\Users\Cinnamon\Desktop\geek_resnet50_upernet\decoder_epoch_5.pth"
    config_path  = r"C:\Users\Cinnamon\Desktop\geek_resnet50_upernet\config.yaml"

    # model initialization
    model = AnimeInference(encoder_path, decoder_path, config_path)

    image_paths = [r"C:\Users\Cinnamon\Desktop\samples\crop\9_crop.png"]
    image_paths = glob.glob(os.path.join(r"C:\Users\Cinnamon\Desktop\samples\crop", '*.png'))

    n_paths = len(image_paths)
    for i, image_path in enumerate(image_paths):
        print ('process %d/%d ...' % (i+1,n_paths))

        # model inference
        results = model.process(image_path)

        # visualize
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask_cls_dict = results[0]
        visualize_image = visualize_pred_mask(image, mask_cls_dict, threshold=0.3)

        #
        output_path = os.path.join(os.path.dirname(image_path), 'segm', os.path.basename(image_path))
        cv2.imwrite(output_path, visualize_image)


if __name__ == '__main__':
    single_test()


