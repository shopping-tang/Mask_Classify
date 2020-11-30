import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models

classify_result = {0 : 'not mask', 1 : 'mask'}


def single_image_test():
    torch.set_grad_enabled(False)
    input_size = 112

    test_image_path = r"F:\demo\mask_classify\data\train_data\1\1_2.jpg"

    trained_model_path = r'F:\demo\mask_classify\net_model\mask_classify.pt'

    net = torch.load(trained_model_path)

    net.eval()
    print('Finished loading model!')

    cudnn.benchmark = True
    device = torch.device("cpu")
    net = net.to(device)

    with torch.no_grad():
        img_raw = cv2.imread(test_image_path)
        img_raw = np.float32(img_raw)

        if (img_raw.shape[0] != input_size) or (img_raw.shape[1] != input_size):
            img_raw = cv2.resize(img_raw, (input_size, input_size))
        input_img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        input_img = input_img / 255

        input_img = input_img.transpose(2, 0, 1)
        input_img = torch.from_numpy(input_img).unsqueeze(0)
        input_img = input_img.to(device)

        result = net(input_img)  # forward pass

        result = torch.max(result, 1)[1]

        result = result.item()
        print("classify_result = ", classify_result[result])

#single_image_test()