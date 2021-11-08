# -*- coding: utf-8 -*-
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.dataset import makeDataLoader
from lib.loss import makeLoss
from lib.models import makeModel
from lib.optimizier import makeOptimizer
from lib.utils import *
import warnings

warnings.filterwarnings("ignore")


def main(
    epoch=200,
    learning_rate=0.5,
    batch_size=4,
    expr_name="capsule_v1",
    model_name="caps",
    is_distributed=False,
    dataset_name="modelnet10",
    use_residual_block=False,
    **kwargs,
):
    model_dir = check_dir(expr_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bw = 32 if kwargs["bandwidths"] is None else kwargs["bandwidths"]
    N, classes, trainloader = makeDataLoader(
        True, bw, batch_size=batch_size, dataset_name=dataset_name, **kwargs
    )
    Ntest, _, testloader = makeDataLoader(
        False, bw, batch_size=batch_size, dataset_name=dataset_name, **kwargs
    )

    # model
    # model_name choce: caps, baseline, resnet
    LAST_EPOCH, model = makeModel(
        model_name,
        model_dir,
        nclasses=len(classes),
        device=device,
        is_distributed=is_distributed,
        use_residual_block=use_residual_block,
        continue_training=kwargs["continue_training"],
    )
    # print(f"Last epoch:{LAST_EPOCH},\nmodel:{model}")
    # loss
    criterion = makeLoss(kwargs["loss"], nclasses=len(classes))
    # optimizer
    optimizer = makeOptimizer(
        kwargs["optimizer"],
        model,
        learning_rate,
        continue_training=kwargs["continue_training"],
        model_dir=model_dir,
    )

    # learning rate
    if kwargs["decay"]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs["decay_step_size"],
            gamma=kwargs["decay_gamma"],
            last_epoch=LAST_EPOCH,
        )
    elif kwargs["exponential"]:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.exp_gamma, last_epoch=LAST_EPOCH
        )
    else:
        scheduler = None

    def train():
        best_acc = 0
        writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        for e in range(LAST_EPOCH + 1, epoch):
            total_correct = 0
            train_loss = 0
            pbar = tqdm(trainloader)
            for i, (X, y) in enumerate(pbar):
                loss, correct = train_step(model, optimizer, criterion, X, y, device, model_name)

                total_correct += correct
                train_loss += loss
                # print(f"[Epoch {e+1}|{i+1}/{BATCHES} Training],loss:{loss},lr:{get_learning_rate(e+1, learning_rate)}")
                des = "[Epoch {}|Training],loss:{:.2f},lr:{:.5f}".format(
                    e + 1, loss, optimizer.param_groups[0]["lr"]
                )
                pbar.set_description(des)
                break
            if scheduler:
                scheduler.step(e)
            # train acc log
            train_acc = total_correct / N
            # test acc log
            result_eval = evaluate(
                model, criterion, testloader, device, model_name, nclasses=len(classes), plot=False
            )
            test_loss, test_acc = result_eval.loss, result_eval.acc
            print("[Epoch {} Train] <ACC>={:2}".format(e + 1, train_acc))
            print(
                "[Epoch {} Test] <ACC>={:2} <LOSS>={:2}".format(
                    e + 1, test_acc, test_loss
                )
            )

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": e,
            }
            # best test acc log
            if test_acc > best_acc:
                # print(f"Get best score after {e+1} epochs, saving model.")
                best_acc = test_acc
                torch.save(state, os.path.join("logs", expr_name, "best_model.ckpt"))
                with open(
                    os.path.join(model_dir, f"{expr_name}_test_acc.txt"), "w"
                ) as f:
                    f.write("{:2}".format(best_acc))

            if e + 1 % 5 == 0:
                torch.save(
                    state, os.path.join("logs", expr_name, f"{model_name}_{e + 1}.ckpt")
                )

            writer.add_scalars(
                f"{expr_name}/loss",
                {"train": train_loss / N, "test": test_loss / Ntest},
                e + 1,
            )
            writer.add_scalars(
                f"{expr_name}/acc", {"train": train_acc, "test": test_acc}, e + 1
            )

            writer.add_scalar(f"{expr_name}/best_acc", best_acc, e + 1)
            writer.close()

    if kwargs["evaluate"]:
        result_eval = evaluate(
            model, criterion, testloader, device, model_name, nclasses=len(classes), plot=False
        )
        test_loss, test_acc, acc_for_every_cls = (
            result_eval.loss,
            result_eval.acc,
            result_eval.acc_for_every_cls,
        )
        print(
            "[{}]\n <total acc>={:2}\n <acc_for_every_cls>={}".format(
                expr_name, test_acc, acc_for_every_cls
            )
        )
    else:
        train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--expr_name", type=str, default="capsule_v1")
    parser.add_argument("--evaluate", default=False, action="store_true")

    # Model options
    parser.add_argument(
        "--model_name",
        type=str,
        default="caps",
        choices=[
            "caps",
            "baseline",
            "resnet",
            "smnist",
            "smnist_baseline",
            "smnist_baseline_deep",
            "msvc",
            "msvc_caps",
        ],
    )
    # whether to use residual block, only for capsule version now
    parser.add_argument(
        "--use_residual_block",
        default=False,
        action="store_true",
        help="use residual block with S2/SO3",
    )


    # Dataset options
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="modelnet10",
        choices=["modelnet10", "modelnet40", "shrec15_0.2","shrec15_0.3","shrec15_0.4","shrec15_0.5","shrec15_0.6","shrec15_0.7", "shrec17", "smnist"],
    )
    parser.add_argument("--type", default="rotate", choices=["rotate", "no_rotate"])
    parser.add_argument("--pick_randomly", default=False, action="store_true")
    # for smnist options
    parser.add_argument("--no_rotate_train", default=False, action="store_true")
    parser.add_argument("--no_rotate_test", default=False, action="store_true")
    parser.add_argument(
        "--overlap",
        default=False,
        action="store_true",
        help="use overlapped data to test",
    )
    # for multi-scale input
    parser.add_argument("--bandwidths", nargs="+", type=int)

    # training options
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--exponential", default=False, action="store_true")
    parser.add_argument("--exp_gamma", type=float, default=0.9)
    parser.add_argument("--decay", default=False, action="store_true")
    parser.add_argument("--decay_step_size", type=int, default=25)
    parser.add_argument("--decay_gamma", type=float, default=0.7)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--loss",
        type=str,
        default="nll",
        choices=["nll", "cross_entropy", "Capsule_recon", "CapsuleLoss" ],
    )
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--continue_training", default=False, action="store_true")
    parser.add_argument(
        "--is_distributed",
        default=False,
        action="store_true",
        help="distributed training",
    )

    args = parser.parse_args()
    print(args)

    main(**args.__dict__)
