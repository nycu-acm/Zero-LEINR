import torch
import torchvision
import os
import glob
import argparse
import datetime
import dataloader
import Myloss
import numpy as np
from torchvision import transforms
from tensorboardX import SummaryWriter

from arch_enhance_net import enhance_net_nopool as EnhanceNet
from arch_unet import UNet as DenoiseNet
from PIL import Image
from tqdm import tqdm
from utils import (
    weights_init,
    generate_mask_pair,
    generate_subimages,
    AugmentNoise,
    calculate_psnr,
    calculate_ssim,
)

def train(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_devices
    # Noise adder
    noise_adder = AugmentNoise(style=config.noisetype)

    ENHANCER_B1 = EnhanceNet().cuda()
    ENHANCER_B2 = EnhanceNet().cuda()
    DENOISER = DenoiseNet()

    if config.parallel:
        DENOISER = torch.nn.DataParallel(DENOISER)
    DENOISER = DENOISER.cuda()

    ENHANCER_B1.apply(weights_init)
    ENHANCER_B2.apply(weights_init)

    if config.load_pretrain_enhance == True:
        ENHANCER_B1.load_state_dict(
            torch.load(
                os.path.join(
                    config.pretrain_model_enhance,
                    "Model_enh_b1",
                    f"Epoch195.pth",
                )
            )
        )
        ENHANCER_B2.load_state_dict(
            torch.load(
                os.path.join(
                    config.pretrain_model_enhance,
                    "Model_enh_b2",
                    f"Epoch195.pth",
                )
            )
        )

    if config.load_pretrain_denoise == True:
        state_dict = torch.load(config.pretrain_model_denoise)
        del state_dict['nlen.enc_conv0.weight']
        del state_dict['nlen.enc_conv0.bias']
        DENOISER.load_state_dict(state_dict, False)

    modules = {
        "ENHANCER_B1": ENHANCER_B1,
        "ENHANCER_B2": ENHANCER_B2,
        "DENOISER": DENOISER,
    }

    # Low quality datasets
    train_dataset = dataloader.lowlight_loader(config.train_dirs, config.patch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    tb_logger_1 = SummaryWriter(log_dir=os.path.join(config.tb_path, "Model_enh_b1"))
    tb_logger_2 = SummaryWriter(log_dir=os.path.join(config.tb_path, "Model_enh_b2"))
    tb_logger_n2n = SummaryWriter(log_dir=os.path.join(config.tb_path, "Model_denoise"))

    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, config.well_exposure)
    L_TV = Myloss.L_TV()
    L_color_angle = Myloss.L_color_angle()
    L_color = Myloss.L_color()

    optimizer_enh_b1 = torch.optim.Adam(ENHANCER_B1.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer_enh_b2 = torch.optim.Adam(ENHANCER_B2.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer_denoise = torch.optim.Adam(DENOISER.parameters(), lr=config.lr_n2n)
    ratio = config.num_epochs / 100
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_denoise,
        milestones=[
            int(20 * ratio) - 1,
            int(40 * ratio) - 1,
            int(60 * ratio) - 1,
            int(80 * ratio) - 1,
        ],
        gamma=config.gamma,
    )

    ENHANCER_B1.train()
    ENHANCER_B2.train()
    DENOISER.train()

    current_step = 0
    highest_psnr = 0.0
    highest_ssim = 0.0
    highest_epoch = 0

    try:
        for epoch in range(config.start_epoch, config.num_epochs):
            for iteration, img_lowlight in enumerate(train_loader):
                current_step += 1
                img_lowlight = img_lowlight.cuda()
                img_lowlight = noise_adder.add_train_noise(img_lowlight)

                assert not torch.isnan(img_lowlight).any(), "Input NAN"
                enhanced_image_mid_1, enhanced_image_1, A_1 = ENHANCER_B1(img_lowlight, "train")
                assert not torch.isnan(A_1).any(), "Alpha Map 1 for Training NAN"
                assert not torch.isnan(enhanced_image_1).any(), "Enhanced Image 1 for Training NAN"

                enhanced_image_mid_2, enhanced_image_2, A_2 = ENHANCER_B2(img_lowlight, "train")
                assert not torch.isnan(A_2).any(), "Alpha Map 2 for Training NAN"
                assert not torch.isnan(enhanced_image_2).any(), "Enhanced Image 2 for Training NAN"

                # illumination smoothness loss
                Loss_TV_1 = 200 * L_TV(A_1)
                assert not (torch.isnan(Loss_TV_1).any() or torch.isinf(Loss_TV_1).any()), "Illumination Smoothness Loss 1 NAN"
                Loss_TV_2 = 200 * L_TV(A_2)
                assert not (torch.isnan(Loss_TV_2).any() or torch.isinf(Loss_TV_2).any()), "Illumination Smoothness Loss 2 NAN"

                # spatial consistency loss
                loss_spa_1 = torch.mean(L_spa(enhanced_image_1, img_lowlight))
                assert not (torch.isnan(loss_spa_1).any() or torch.isinf(loss_spa_1).any()), "Spatial Consistency Loss 1 NAN"
                loss_spa_2 = torch.mean(L_spa(enhanced_image_2, img_lowlight))
                assert not (torch.isnan(loss_spa_2).any() or torch.isinf(loss_spa_2).any()), "Spatial Consistency Loss 2 NAN"

                # color constancy loss
                loss_col = 5 * torch.mean(L_color(enhanced_image_1))
                assert not (torch.isnan(loss_col).any() or torch.isinf(loss_col).any()), "Color Constancy Loss NAN"

                # exposure control loss
                loss_exp_1 = 10 * torch.mean(L_exp(enhanced_image_1))
                assert not (torch.isnan(loss_exp_1).any() or torch.isinf(loss_exp_1).any()), "Exposure Control Loss 1 NAN"
                loss_exp_2 = 10 * torch.mean(L_exp(enhanced_image_2))
                assert not (torch.isnan(loss_exp_2).any() or torch.isinf(loss_exp_2).any()), "Exposure Control Loss 2 NAN"

                # color angle loss
                loss_color_angle = 0.5 * L_color_angle(img_lowlight, enhanced_image_2)
                assert not (torch.isnan(loss_color_angle).any() or torch.isinf(loss_color_angle).any()), "Color Angle Loss NAN"

                # Optimize first Zero-DCE
                loss_1 = Loss_TV_1 + loss_spa_1 + loss_col + loss_exp_1
                optimizer_enh_b1.zero_grad()
                loss_1.backward()
                torch.nn.utils.clip_grad_norm(ENHANCER_B1.parameters(), config.grad_clip_norm)
                optimizer_enh_b1.step()

                # Optimize second Zero-DCE
                loss_2 = Loss_TV_2 + loss_spa_2 + loss_color_angle + loss_exp_2
                optimizer_enh_b2.zero_grad()
                loss_2.backward()
                torch.nn.utils.clip_grad_norm(ENHANCER_B2.parameters(), config.grad_clip_norm)
                optimizer_enh_b2.step()

                # Optimize Fusion network
                with torch.no_grad():
                    enhanced_image_mid_1, enhanced_image_1, A_1 = ENHANCER_B1(img_lowlight, "train")
                    assert not torch.isnan(A_1).any(), "Alpha Map 1 for Fusion NAN"
                    assert not torch.isnan(enhanced_image_1).any(), "Enhanced Image 1 for Fusion NAN"

                    enhanced_image_mid_2, enhanced_image_2, A_2 = ENHANCER_B2(img_lowlight, "train")
                    assert not torch.isnan(A_2).any(), "Alpha Map 2 for Fusion NAN"
                    assert not torch.isnan(enhanced_image_2).any(), "Enhanced Image 2 for Fusion NAN"

                    # fusion
                    direction = torch.nn.functional.normalize(enhanced_image_2, dim=1)
                    magnitude = torch.norm(enhanced_image_1, dim=1).unsqueeze(dim=1)
                    fused = magnitude * direction

                noisy = fused

                optimizer_denoise.zero_grad()

                # DENOISER Part

                mask1, mask2 = generate_mask_pair(noisy)
                noisy_sub1 = generate_subimages(noisy, mask1)
                noisy_sub2 = generate_subimages(noisy, mask2)
                with torch.no_grad():
                    noisy_denoised = DENOISER(noisy)
                noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
                noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

                noisy_output = DENOISER(noisy_sub1)
                noisy_target = noisy_sub2
                Lambda = epoch / config.num_epochs * config.increase_ratio
                diff = noisy_output - noisy_target
                exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

                loss_rec = torch.mean(diff**2)
                assert not (torch.isnan(loss_rec).any() or torch.isinf(loss_rec).any()), print(loss_rec)
                loss_reg = Lambda * torch.mean((diff - exp_diff) ** 2)
                assert not (torch.isnan(loss_reg).any() or torch.isinf(loss_reg).any()), print(loss_reg)
                
                loss_denoise = config.Lambda1 * loss_rec + config.Lambda2 * loss_reg

                loss_denoise.backward()

                optimizer_denoise.step()

                # # Tensorboard for model 1
                tb_logger_1.add_scalar("L_TV_1", Loss_TV_1.item(), current_step)
                tb_logger_1.add_scalar("L_spa_1", loss_spa_1.item(), current_step)
                tb_logger_1.add_scalar("L_col", loss_col.item(), current_step)
                tb_logger_1.add_scalar("L_exp_1", loss_exp_1.item(), current_step)
                tb_logger_1.add_scalar("Total_Loss_enh_b1", loss_1.item(), current_step)

                # Tensorboard for model 2
                tb_logger_2.add_scalar("L_TV_2", Loss_TV_2.item(), current_step)
                tb_logger_2.add_scalar("L_spa_2", loss_spa_2.item(), current_step)
                tb_logger_2.add_scalar("L_exp_2", loss_exp_2.item(), current_step)
                tb_logger_2.add_scalar("L_color_angle", loss_color_angle.item(), current_step)
                tb_logger_2.add_scalar("Total_Loss_enh_b2", loss_2.item(), current_step)

                # Tensorboard for DENOISER
                tb_logger_n2n.add_scalar("L_rec", loss_rec.item(), current_step)
                tb_logger_n2n.add_scalar("L_reg", loss_reg.item(), current_step)
                tb_logger_n2n.add_scalar("Total_Loss_denoise", loss_denoise.item(), current_step)

                if ((iteration + 1) % config.display_iter) == 0:
                    print(
                        "Loss_enh_b1 at epoch {:d} iteration {:d}".format(epoch, iteration + 1),
                        ":",
                        loss_1.item(),
                    )
                    print(
                        "Loss_enh_b2 at epoch {:d} iteration {:d}".format(epoch, iteration + 1),
                        ":",
                        loss_2.item(),
                    )
                    print(
                        "Loss_denoise at epoch {:d} iteration {:d}".format(epoch, iteration + 1),
                        ":",
                        loss_denoise.item(),
                    )

                if ((iteration + 1) % config.snapshot_iter) == 0:
                    torch.save(
                        ENHANCER_B1.state_dict(),
                        os.path.join(
                            config.snapshot_path,
                            "Model_enh_b1",
                            "Epoch" + str(epoch) + ".pth",
                        ),
                    )
                    torch.save(
                        ENHANCER_B2.state_dict(),
                        os.path.join(
                            config.snapshot_path,
                            "Model_enh_b2",
                            "Epoch" + str(epoch) + ".pth",
                        ),
                    )
                    torch.save(
                        DENOISER.state_dict(),
                        os.path.join(
                            config.snapshot_path,
                            "Model_denoise",
                            "Epoch" + str(epoch) + ".pth",
                        ),
                    )
            scheduler.step()

            avg_psnr, avg_ssim = validate(modules, config)

            if avg_psnr * 0.5 + avg_ssim * 0.5 > highest_psnr * 0.5 + highest_ssim * 0.5:
                highest_psnr = avg_psnr
                highest_ssim = avg_ssim
                highest_epoch = epoch
            print(f"PSNR/SSIM at epoch {epoch}: {avg_psnr}/{avg_ssim}")
            log_path = os.path.join(config.result_path, "result.csv")
            with open(log_path, "a") as f:
                f.writelines("{},{},{}\n".format(epoch, avg_psnr, avg_ssim))
        test(modules, config)
    except AssertionError as e:
        raise


def validate(modules, config):
    with torch.no_grad():
        file_path = config.val_input_dirs
        gt_path = config.val_gt_dirs

        images = glob.glob(file_path + "/*")

        avg_psnr = 0
        avg_ssim = 0
        cnt = 0

        for image in images:
            image_name = image.split("/")[-1]

            input_image = Image.open(os.path.join(file_path, image_name))
            input_image = np.asarray(input_image) / 255.0
            input_image = torch.from_numpy(input_image).float()
            input_image = input_image.permute(2, 0, 1)
            input_image = input_image.cuda().unsqueeze(0)

            gt = Image.open(os.path.join(gt_path, image_name))
            gt = np.array(gt).astype(np.uint8)

            _, enhanced_image_1, A_1 = modules["ENHANCER_B1"](input_image, "val")
            _, enhanced_image_2, A_2 = modules["ENHANCER_B2"](input_image, "val")

            direction = torch.nn.functional.normalize(enhanced_image_2, dim=1)
            magnitude = torch.norm(enhanced_image_1, dim=1).unsqueeze(dim=1)
            fused = magnitude * direction

            im = fused.permute(0, 2, 3, 1)
            im = im.cpu().data.clamp(0, 1).numpy()
            im = im.squeeze()
            im = np.clip(im * 255.0 + 0.5, 0, 255).astype(np.uint8)

            H = im.shape[0]
            W = im.shape[1]
            val_size = (max(H, W) + 31) // 32 * 32
            im = np.pad(im, [[0, val_size - H], [0, val_size - W], [0, 0]], "reflect")
            transformer = transforms.Compose([transforms.ToTensor()])
            im = transformer(im)
            im = torch.unsqueeze(im, 0)
            im = im.cuda()

            prediction = modules["DENOISER"](im)
            prediction = prediction[:, :, :H, :W]
            prediction = prediction.cpu().data.clamp(0, 1).numpy()
            prediction = prediction.squeeze()
            prediction = np.clip(prediction * 255.0 + 0.5, 0, 255).astype(np.uint8)
            prediction = np.transpose(prediction, [1, 2, 0])

            avg_psnr += calculate_psnr(prediction, gt)
            avg_ssim += calculate_ssim(prediction, gt)

            cnt += 1
    avg_psnr /= cnt
    avg_ssim /= cnt

    return avg_psnr, avg_ssim


def test(modules, config):
    with torch.no_grad():
        test_path = config.test_dirs
        included_datasets = [
            "LOL_dataset",
            # "LIME",
            # "MEF",
            # "NPE",
            # "VV",
        ]
        dataset_list = os.listdir(test_path)
        print(dataset_list)

        for dataset_name in dataset_list:
            if dataset_name not in included_datasets:
                continue

            if dataset_name == "LOL_dataset":
                test_list = glob.glob(test_path + "/" + dataset_name + "/eval15/low/*")
            else:
                test_list = glob.glob(test_path + "/" + dataset_name + "/*")

            for image in tqdm(test_list):
                image_path = image
                os.makedirs(os.path.join(config.result_path, dataset_name), exist_ok=True)

                data_lowlight = Image.open(image).convert("RGB")

                data_lowlight = np.asarray(data_lowlight) / 255.0

                data_lowlight = torch.from_numpy(data_lowlight).float()
                data_lowlight = data_lowlight.permute(2, 0, 1)
                data_lowlight = data_lowlight.cuda().unsqueeze(0)

                _, enhanced_image_1, A_1 = modules["ENHANCER_B1"](data_lowlight, "val")
                _, enhanced_image_2, A_2 = modules["ENHANCER_B2"](data_lowlight, "val")

                direction = torch.nn.functional.normalize(enhanced_image_2, dim=1)
                magnitude = torch.norm(enhanced_image_1, dim=1).unsqueeze(dim=1)
                fused = magnitude * direction

                im = fused.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze()
                im = np.clip(im * 255.0 + 0.5, 0, 255).astype(np.uint8)

                H = im.shape[0]
                W = im.shape[1]
                val_size = (max(H, W) + 31) // 32 * 32
                im = np.pad(im, [[0, val_size - H], [0, val_size - W], [0, 0]], "reflect")
                transformer = transforms.Compose([transforms.ToTensor()])
                im = transformer(im)
                im = torch.unsqueeze(im, 0)
                im = im.cuda()

                prediction = modules["DENOISER"](im)
                prediction = prediction[:, :, :H, :W]

                if not os.path.exists(image_path.replace("/" + image_path.split("/")[-1], "")):
                    os.makedirs(image_path.replace("/" + image_path.split("/")[-1], ""))
                imgname_list = image_path.split("/")[-1].split(".")

                torchvision.utils.save_image(
                    fused,
                    os.path.join(
                        config.result_path,
                        dataset_name,
                        imgname_list[0] + "_fused." + imgname_list[-1],
                    ),
                )
                torchvision.utils.save_image(
                    prediction,
                    os.path.join(
                        config.result_path,
                        dataset_name,
                        imgname_list[0] + "_denoised." + imgname_list[-1],
                    ),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_name", default="unet", type=str)
    parser.add_argument("--gpu_devices", default="0", type=str)
    parser.add_argument(
        "--train_dirs",
        type=str,
        default="/mnt/HDD3/wingho/datasets/fivek_zerodce_lol/",
    )
    parser.add_argument("--noisetype", type=str, default="poisson5_50")
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--lr_n2n", type=float, default=1e-4)
    parser.add_argument("--well_exposure", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--increase_ratio", type=float, default=2.0)
    parser.add_argument("--n_feature_n2n", type=int, default=48)
    parser.add_argument("--n_channel_n2n", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--Lambda1", type=float, default=1.0)
    parser.add_argument("--Lambda2", type=float, default=1.0)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--display_iter", type=int, default=10)
    parser.add_argument("--snapshot_iter", type=int, default=10)
    parser.add_argument("--snapshots_folder", type=str, default="snapshots/")
    parser.add_argument("--tensorboard_folder", type=str, default="tb_logger/")
    parser.add_argument("--results_folder", type=str, default="results")
    parser.add_argument("--val_input_dirs", type=str, default="/mnt/HDD3/wingho/datasets/LOL_dataset/eval15/low/")
    parser.add_argument("--val_gt_dirs", type=str, default="/mnt/HDD3/wingho/datasets/LOL_dataset/eval15/high/")
    parser.add_argument("--test_dirs", type=str, default="/mnt/HDD3/wingho/datasets")
    parser.add_argument("--load_pretrain_enhance", action="store_true")
    parser.add_argument("--load_pretrain_denoise", action="store_true")
    parser.add_argument("--pretrain_model_enhance", type=str, default="")
    parser.add_argument("--pretrain_model_denoise", type=str, default="")
    config = parser.parse_args()

    systime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    snapshot_path = os.path.join(config.snapshots_folder, config.network_name, systime)
    tb_path = os.path.join(config.tensorboard_folder, config.network_name, systime)

    config.snapshot_path = snapshot_path
    config.tb_path = tb_path
    config.result_path = os.path.join(config.results_folder, config.network_name, systime)

    os.makedirs(os.path.join(snapshot_path, "Model_enh_b1"), exist_ok=True)
    os.makedirs(os.path.join(snapshot_path, "Model_enh_b2"), exist_ok=True)
    os.makedirs(os.path.join(snapshot_path, "Model_denoise"), exist_ok=True)
    os.makedirs(os.path.join(tb_path, "Model_enh_b1"), exist_ok=True)
    os.makedirs(os.path.join(tb_path, "Model_enh_b2"), exist_ok=True)
    os.makedirs(os.path.join(tb_path, "Model_denoise"), exist_ok=True)
    os.makedirs(os.path.join(config.result_path), exist_ok=True)

    train(config)
