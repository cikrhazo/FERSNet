import argparse
import torch.utils.data as data
from network import *
import torch.optim as optim
from criteria import *
from data import MakeDataSet
from torch.autograd import Variable
import os, sys, time
import shutil
import numpy as np
import logging
import cv2
from mmcv.utils import get_logger


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(args):
    beginner = args.beginner
    stride = args.stride
    fold = args.fold
    device = args.device
    batch_size = args.batch_size
    epoch = args.Epoch
    lr = args.lr
    weight_decay = args.weight_decay
    model_path = './models/' + str(fold).zfill(2)
    mem_size = args.mem_size
    kernel_size = args.kernel_size
    num_class = args.num_class
    img_size = args.image_size
    img_ch = args.image_channel
    emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    if os.path.exists(model_path) is not True:
        os.makedirs(model_path)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join("./logs/", f'{timestamp}.log')
    logger = get_logger(name='Multitask FER', log_file=log_file, log_level=logging.INFO)
    logger.info(f"batch size {batch_size}")

    torch.cuda.set_device(device)
    
    TrainSet = MakeDataSet(root=args.train_root, train=True, out_size=img_size, fold=fold)
    TrainLoader = data.DataLoader(TrainSet, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    ValidSet = MakeDataSet(root=args.train_root, train=False, out_size=img_size, fold=fold)
    ValidLoader = data.DataLoader(ValidSet, batch_size=6, shuffle=False, pin_memory=False, num_workers=2)
    iters = int(TrainSet.__len__() / batch_size * (epoch - beginner))

    net = FERSNet(vgg_name='VGG13', num_class=num_class, mem_size=mem_size, k_channel=img_ch)
    if args.load_checkpoint:
        logger.info("loading checkpoint from" + ' ./models/net_'+str(beginner).zfill(3)+'.pth')
        net.load_state_dict(torch.load(
            './models/net_'+str(beginner).zfill(3)+'.pth', map_location=torch.device(device)
        ))
    # else:
    #     logger.info("loading checkpoint from" + ' ./models/init_model/pretrain.pth')
    #     checkpoint = torch.load(
    #         './models/init_model/pretrain.pth', map_location=torch.device(device)
    #     )
    #     pretrained_state_dict = {k: v for k, v in checkpoint.items() if "decoder" not in k and "transform" not in k}
    #     model_state_dict = net.state_dict()
    #     model_state_dict.update(pretrained_state_dict)
    #     net.load_state_dict(model_state_dict)

    optimizer_C = optim.Adam(net.parameters(), weight_decay=weight_decay, betas=(0.5, 0.999), lr=lr)
    scheduler_C = optim.lr_scheduler.MultiStepLR(optimizer_C, milestones=[int(e) for e in args.milestones.split(',')],
                                                 gamma=args.gamma)
    
#     scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, eta_min=1e-5,
#                                                        T_max=int(TrainSet.__len__() / batch_size * epoch))

    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.cuda(device=device)

    dis = Discriminator(input_shape=(img_ch, img_size, img_size), num_class=num_class)
    optimizer_D = optim.Adam(dis.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.5, 0.999))
    scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[int(e) for e in args.milestones.split(',')],
                                                 gamma=args.gamma)
    
#     scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, eta_min=1e-5,
#                                                        T_max=int(TrainSet.__len__() / batch_size * epoch))

    dis = torch.nn.DataParallel(dis, device_ids=[0, 1])
    dis.cuda(device=device)

    loss_fc_cls = cls_criterion().cuda()
    loss_fc_rec = rec_criterion().cuda()
    loss_fc_dcn = dcn_criterion().cuda()
    
    print("Testing Subjects: " + ' '.join(TrainSet.subject_list))
    print("# Training Samples: " + str(TrainSet.__len__()))

    BestValidAccuracy = 0

    for i in range(beginner, epoch):

        tra_loss = 0
        cls_loss = 0
        idn_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        print('Training FESRNet... with learning rate: %.4e' % optimizer_C.param_groups[0]['lr'])

        for batch_idx, (face, label, face_target, label_target) in enumerate(TrainLoader):

            iterPerEpoch += 1
            trainSamples += face.size(0)

            face = Variable(face.cuda(device), requires_grad=False)
            label = Variable(label.cuda(device), requires_grad=False)
            face_target = Variable(face_target.cuda(device), requires_grad=False)
            label_target = Variable(label_target.cuda(device), requires_grad=False)

            prob_s = torch.zeros((label.size(0), num_class), dtype=torch.float32).to(face.device)
            prob_s = prob_s.scatter_(1, label.data, 1.)

            prob_t = torch.zeros((label_target.size(0), num_class), dtype=torch.float32).to(face.device)
            prob_t = prob_t.scatter_(1, label_target.data, 1.)
            cond_t = prob_t.unsqueeze(2).unsqueeze(3).repeat(1, 1, img_size, img_size)

            valid = Variable(torch.ones(size=(face.size(0), *dis.module.out_size)).cuda(device), requires_grad=False)
            fake = Variable(torch.zeros(size=(face.size(0), *dis.module.out_size)).cuda(device), requires_grad=False)

            net.train(False)
            dis.train(True)
            with torch.no_grad():
                face_syn_, _ = net(face, prob_t)
            face_syn_.detach()
            # face_syn_ = torch.clamp(face_syn_, min=0, max=1)

            for _ in range(3):
                optimizer_D.zero_grad()
                validity_real = dis(face_target, cond_t)
                error_real = loss_fc_dcn(validity_real, valid)
                validity_fake = dis(face_syn_, cond_t)
                error_fake = loss_fc_dcn(validity_fake, fake)
                error = (error_real + error_fake) / 2.
                error.backward()
                optimizer_D.step()

            net.train(True)
            dis.train(False)

            optimizer_C.zero_grad()
            face_syn, predict_org = net(face, prob_t)
            face_cyc, _ = net(face_syn, prob_s)
            Test_fake = dis(face_syn, cond_t)

            loss_cls_ = loss_fc_cls(predict_org, label.squeeze(1))
            loss_idn_ = loss_fc_rec(face_syn, face_target)
            loss_cyc_ = nn.MSELoss()(face_cyc, face)
            loss_tra_ = loss_fc_dcn(Test_fake, valid)
            loss_all = loss_cls_ + loss_idn_ + 0.3*loss_tra_ + 0.5 * loss_cyc_
            loss_all.backward()

            optimizer_C.step()

            label_t = label.detach().squeeze().cuda(device)
            _, label_p = torch.max(predict_org, 1)
            numCorrTrain += label_p.eq(label_t.data).cpu().sum()

            cls_loss += loss_cls_.cpu().detach().numpy().astype(float)
            tra_loss += loss_tra_.cpu().detach().numpy().astype(float)
            idn_loss += loss_idn_.cpu().detach().numpy().astype(float)
            if batch_idx % 10 == 9 or batch_idx == 0:
                print('#batch: %3d; loss_cls: %f; loss_tran: %f; loss_idn: %f;'
                      % (batch_idx + 1, loss_cls_, loss_tra_, loss_idn_))

        avg_tra = tra_loss / iterPerEpoch
        avg_cls = cls_loss / iterPerEpoch
        avg_idn = idn_loss / iterPerEpoch
        trainAccuracy = (int(numCorrTrain) / trainSamples) * 100
        logger.info('Train Epoch = {} | Tra Loss = {} | Idn Loss = {} | Cls Loss = {} | Train Accuracy = {}'
              .format(i + 1, round(avg_tra, 4), round(avg_idn, 4), round(avg_cls, 4), round(trainAccuracy, 4)))

        scheduler_C.step()
        scheduler_D.step()

        # Validation:
        net.train(False)
        validSamples = 0
        numCorrValid = 0
        for idx, (face_valid, label_valid) in enumerate(ValidLoader):
            face_valid = Variable(face_valid.cuda(), requires_grad=False)
            label_valid = Variable(label_valid.cuda(), requires_grad=False)
            k = face_valid.size(0)
            trg_label = torch.randint(low=0, high=num_class, size=(k, 1))
            prob_t = torch.zeros((k, num_class), dtype=torch.float32).to(face_valid.device)
            prob_t = prob_t.scatter_(1, trg_label.to(face_valid.device), 1.)
            with torch.no_grad():
                out_tensor, prob = net(face_valid, prob_t)

            label_t = label_valid.detach().squeeze().to(face_valid.device)
            _, label_p = torch.max(prob, 1)
            numCorrValid += (label_p == label_t.squeeze()).sum()
            validSamples += label_t.size(0)

            if idx == 0:
                shutil.rmtree('./samples')
                os.mkdir('./samples')
                for ii in range(out_tensor.size(0)):
                    label_src = float(label_valid[ii, 0].detach().cpu())
                    label_src = emotion_list[int(label_src)]
                    label_tar = float(trg_label[ii, 0].detach().cpu())
                    label_tar = emotion_list[int(label_tar)]
                    img_syn = out_tensor[ii, :, :, :].detach().cpu().numpy()
                    img_syn = img_syn.transpose((1, 2, 0))
                    img_syn = np.uint8(img_syn.clip(0, 1) * 255)
                    cv2.imwrite("./samples/image_%d_" % ii + label_src+"To"+label_tar+".png", img_syn)

        validAccuracy = (int(numCorrValid) / validSamples) * 100

        if BestValidAccuracy <= validAccuracy:
            BestValidAccuracy = validAccuracy
            torch.save(net.module.state_dict(), os.path.join(model_path, 'FERSNet_' + str(fold).zfill(2) + '.pth'))

        logger.info('Train Epoch = {} | Valid Accuracy = {} | Best Accuracy = {}'
              .format(i + 1, round(validAccuracy, 4), round(BestValidAccuracy, 4)))

        if i % stride == (stride - 1):
            torch.save(net.module.state_dict(), os.path.join(model_path, 'net_' + str(i + 1).zfill(3) + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For Dataset and Record
    parser.add_argument("--train_root", type=str, default='/media/ruizhao/programs/datasets/Face/MMI/MTFER/')
    parser.add_argument("--image_size", type=int, default=96, help="width and height should be identical")
    parser.add_argument("--image_channel", type=int, default=3)
    parser.add_argument("--stride", type=int, default=50, help='the stride for saving models')
    parser.add_argument("--fold", type=int, default=2, help="# of the fold of the 10 fold cross-validation")
    # For Network
    parser.add_argument("--num_class", type=int, default=6)
    parser.add_argument("--mem_size", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=3, help="pooling kernel and anti-pooling kernel")
    # For Training
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--beginner", type=int, default=0)
    parser.add_argument('--Epoch', type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--milestones', default='150, 400', type=str)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    main(args)
