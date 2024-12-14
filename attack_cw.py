import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from matplotlib import cm
from tqdm import tqdm

from alg.vae_new import bayes_classifier, construct_optimizer
from utils.utils import load_data, load_model


def perform_attacks(
    data_name, epsilons, batch_size, save_dir="./results/", device=None
):
    """
    Perform FGSM attack on a given model, evaluate accuracy vs. epsilon, and save results.

    Args:
        data_name (str): Dataset name ('mnist').
        epsilons (list): List of epsilon values for FGSM attack.
        batch_size (int): Batch size for evaluating the model.
        save_dir (str): Directory to save results.
        device (torch.device, optional): Device to load the model onto.
    """
    vae_types = ["A", "B", "C", "D", "E", "F", "G"]

    attack_methods = ["CW"]
    accuracies = {vae_type: {} for vae_type in vae_types}
    for vae_type in vae_types:
        encoder, generator = load_model(data_name, vae_type, 0, device=device)
        encoder.eval()
        input_shape = (1, 28, 28) if data_name == "mnist" else (3, 32, 32)
        dimY = 10 if data_name != "gtsrb" else 43
        ll = "l2"
        K = 10

        if vae_type == "A":
            dec = (generator.pyz_params, generator.pxzy_params)
            from alg.lowerbound_functions import lowerbound_A as lowerbound
        elif vae_type == "B":
            dec = (generator.pzy_params, generator.pxzy_params)
            from alg.lowerbound_functions import lowerbound_B as lowerbound
        elif vae_type == "C":
            dec = (generator.pyzx_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_C as lowerbound
        elif vae_type == "D":
            dec = (generator.pyzx_params, generator.pzx_params)
            from alg.lowerbound_functions import lowerbound_D as lowerbound
        elif vae_type == "E":
            dec = (generator.pyz_params, generator.pzx_params)
            from alg.lowerbound_functions import lowerbound_E as lowerbound
        elif vae_type == "F":
            dec = (generator.pyz_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_F as lowerbound
        elif vae_type == "G":
            dec = (generator.pzy_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_G as lowerbound
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")

        enc_conv = encoder.encoder_conv
        enc_mlp = encoder.enc_mlp
        enc = (enc_conv, enc_mlp)
        model = lambda x: bayes_classifier(
            x,
            (enc_conv, enc_mlp),
            dec,
            ll,
            dimY,
            lowerbound=lowerbound,
            K=10,
            beta=1.0,
        )[0]
        X_ph = torch.zeros(1, *input_shape).to(next(encoder.parameters()).device)
        Y_ph = torch.zeros(1, dimY).to(next(encoder.parameters()).device)
        _, eval_fn = construct_optimizer(X_ph, Y_ph, enc, dec, ll, K, vae_type)

        _, test_dataset = load_data(data_name, path="./data", labels=None, conv=True)
        # subset for debug
        test_dataset = torch.utils.data.Subset(test_dataset, range(10))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        os.makedirs(save_dir, exist_ok=True)

        accuracies[vae_type] = {attack: [] for attack in attack_methods}

        for epsilon in epsilons:
            for attack in attack_methods:
                x_adv = []
                x_clean = []
                y_adv = []
                y_adv_logits = []
                y_clean = []
                y_clean_logits = []
                correct = 0
                total = 0
                pbar = tqdm(total=len(test_loader), desc=f"{attack}, c={epsilon}")
                for images, labels in test_loader:
                    images, labels = images.to(
                        next(encoder.parameters()).device
                    ), labels.to(next(encoder.parameters()).device)

                    if attack == "CW":
                        adv_images = carlini_wagner_l2(
                            model,
                            images,
                            n_classes=dimY,
                            targeted=False,
                            initial_const=epsilon,
                            max_iterations=1000,
                            binary_search_steps=1,
                        )
                    else:
                        raise ValueError(f"Unsupported attack: {attack}")

                    y_pred_clean, y_pred_clean_logit = bayes_classifier(
                        images,
                        (enc_conv, enc_mlp),
                        dec,
                        ll,
                        dimY,
                        lowerbound=lowerbound,
                        K=10,
                        beta=1.0,
                    )

                    y_pred_adv, y_pred_adv_logit = bayes_classifier(
                        adv_images,
                        (enc_conv, enc_mlp),
                        dec,
                        ll,
                        dimY,
                        lowerbound=lowerbound,
                        K=10,
                        beta=1.0,
                    )
                    correct += (torch.argmax(y_pred_adv, dim=1) == labels).sum().item()
                    total += labels.size(0)
                    pbar.update(1)

                    # Save batch data
                    x_adv.append(adv_images.detach().cpu())
                    y_adv.append(y_pred_adv.detach().cpu())
                    y_adv_logits.append(y_pred_adv_logit.detach().cpu())
                    x_clean.append(images.detach().cpu())
                    y_clean.append(y_pred_clean.detach().cpu())
                    y_clean_logits.append(y_pred_clean_logit.detach().cpu())

                # Concatenate and save results
                x_adv = torch.cat(x_adv, dim=0)
                y_adv = torch.cat(y_adv, dim=0)
                y_adv_logits = torch.cat(y_adv_logits, dim=0)
                x_clean = torch.cat(x_clean, dim=0)
                y_clean = torch.cat(y_clean, dim=0)
                y_clean_logits = torch.cat(y_clean_logits, dim=0)

                save_path = os.path.join(
                    save_dir, f"{vae_type}_{attack}_{data_name}_c_{epsilon}.pkl"
                )
                results = {
                    "x_adv": x_adv,
                    "y_adv": y_adv,
                    "y_adv_logits": y_adv_logits,
                    "x_clean": x_clean,
                    "y_clean": y_clean,
                    "y_clean_logits": y_clean_logits,
                }
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)
                print(f"Adversarial examples saved at {save_path}.")
                accuracies[vae_type][attack].append(correct / total)
                print(
                    f"Param value: {epsilon}, VAE type: {vae_type}, Attack: {attack}, Accuracy: {correct / total}"
                )
                pbar.close()

    with open(
        os.path.join(save_dir, f"{data_name}_accuracy_vs_epsilon.json"), "w"
    ) as f:
        json.dump(accuracies, f)

    return accuracies


def plot_results(json_file, save_dir, data_name, epsilons):
    with open(json_file, "r") as f:
        accuracies = json.load(f)
    vae_types = list(accuracies.keys())
    attack_methods = list(accuracies[vae_types[0]].keys())
    letter_to_title = {
        "A": "GFZ",
        "B": "GFY",
        "C": "GBZ",
        "D": "GBY",
        "E": "DFX",
        "F": "DFZ",
        "G": "DBX",
    }
    fig, axes = plt.subplots(1, len(attack_methods), figsize=(12, 4))
    num_vae_types = len(vae_types)
    cmap = cm.get_cmap("rainbow", num_vae_types)
    for i, attack in enumerate(attack_methods):
        for j, vae_type in enumerate(vae_types):
            color = cmap(j)
            axes[i].plot(
                epsilons,
                accuracies[vae_type][attack],
                marker="o",
                label=letter_to_title[vae_type],
                linewidth=2,
                color=color,
            )
        axes[i].set_title(f"{attack} victim acc")
        axes[i].set_xlabel("c")
        axes[i].grid(linestyle="--")
        axes[i].legend()

    # Save the plot
    plt.tight_layout()
    filename = f"{data_name}__accuracy_vs_c_combined.png"
    plt.savefig(
        os.path.join(
            save_dir,
            filename,
        )
    )
    plt.close()
    print(
        "Saved combined accuracy plot for all attacks to",
        os.path.join(save_dir, filename),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform FGSM attack on a trained VAE."
    )

    parser.add_argument(
        "--data_name",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "gtsrb"],
        help="Dataset name (e.g., 'mnist').",
    )

    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0, 0.1, 1, 10, 100, 1000],
        help="List of epsilon values for FGSM attack.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for evaluating the model.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./results/", help="Directory to save results."
    )
    parser.add_argument("--compute", action="store_true", help="Compute the results.")
    parser.add_argument(
        "--plot", action="store_true", help="Plot the results from the JSON file."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="JSON file containing the results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to load the model onto (e.g., 'cuda:0').",
    )

    args = parser.parse_args()
    if args.compute:
        results = perform_attacks(
            data_name=args.data_name,
            epsilons=args.epsilons,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            device=args.device,
        )
    if args.plot:
        if args.json_file is None:
            print("No JSON file specified.")
            args.json_file = os.path.join(
                args.save_dir, f"{args.data_name}_accuracy_vs_epsilon.json"
            )
        plot_results(args.json_file, args.save_dir, args.data_name, args.epsilons)
