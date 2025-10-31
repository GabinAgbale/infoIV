import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from PIL import Image

def linear_regression(X, Y):
    """
    Perform linear regression using PyTorch.

    Parameters
    ----------
    X : torch.Tensor
        Input data.
    Y : torch.Tensor
        Target data.

    Returns
    -------
    torch.Tensor
        The coefficients of the linear regression model.
    """
    device = X.device
    
    X = torch.cat([X, torch.ones(X.shape[0], 1).to(device)], dim=1)
    
    coeffs = torch.linalg.pinv(X) @ Y
    
    W, b = coeffs[:-1], coeffs[-1]
    return W, b

def log_mse_vs_shift_plot(shifts, mses, logger, global_step=0, tag="MSE_vs_Shift"):
    """
    Plots and logs MSE vs. shift using Seaborn and TensorBoard.

    Args:
        shifts (List[float]): Shift values (x-axis).
        mses (List[float]): Corresponding MSE values (y-axis).
        logger (TensorBoardLogger): Lightning logger instance (e.g., self.logger).
        global_step (int): Global step (e.g., epoch) for logging.
        tag (str): Image tag name in TensorBoard.
    """
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=shifts, y=mses, marker="o")
    plt.xlabel("Shift")
    plt.ylabel("MSE")
    plt.title("MSE vs. Shift")
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Log to TensorBoard
    if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
        logger.experiment.add_image(tag, image_tensor[0], global_step)
        print(f"Logged {tag} at step {global_step}")

    plt.close()


def log_dim1_plot(dataset, latent_model, causal_model, logger, global_step=0):

    Y_true = dataset.Y
    Z_true = dataset.Z
    Y_true, Z_true = Y_true.cpu(), Z_true.cpu()
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=Z_true[:, 0].numpy(), y=Y_true[:, 0].numpy(), label="Observed", color="grey", )
    
    # True causal effect
    z_range = torch.linspace(Z_true.min(), Z_true.max(), 100).unsqueeze(1)
    y_true = dataset.l(z_range)

    # Predict using the causal model
    z_pred = latent_model.encoder(dataset.X)
    W, b = linear_regression(dataset.A, z_pred)
    v_pred = z_pred - dataset.A @ W.T - b
    y_pred = causal_model(z_pred, v_pred)

    sns.lineplot(x=z_pred.detach().cpu().numpy().squeeze(), y=y_pred.detach().cpu().numpy().squeeze(), label="Predicted", color="orange")
    sns.lineplot(x=z_range.squeeze(), y=y_true.detach().numpy().squeeze(), label="True Causal Effect", color="blue")

    plt.xlabel("Treatment")
    plt.ylabel("Effect")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Log to TensorBoard
    if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
        logger.experiment.add_image("Low-dimension Causal Effect Estimation", image_tensor[0], global_step)

    plt.close()

def log_pred_gt_plot(y_pred, y_gt, logger, global_step=0, tag="Y pred vs g-t", counter=0):
    y_pred, y_gt = y_pred[:, 0].cpu().numpy(), y_gt[:, 0].cpu().numpy()

    min_val = min(y_pred.min(), y_gt.min())
    max_val = max(y_pred.max(), y_gt.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

    sns.scatterplot(x=y_pred, y=y_gt, color="blue", alpha=.5)
    plt.xlabel("Estimated Y")
    plt.xlabel("Ground-truth Y")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Log to TensorBoard
    if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
        logger.experiment.add_image(f"Estimation v G-T {counter}", image_tensor[0], global_step)
        print(f"Logged {tag} at step {global_step}")
    plt.close()


def log_pred_gt_multiplot(pred_gt_list, logger, global_step=0, tag="Y pred vs g-t"):
    """
    pred_gt_list: List of (y_pred, y_gt) tuples, one per shift. Each is a torch tensor of shape (N, 1)
    """
    num_shifts = len(pred_gt_list)
    cols = min(3, num_shifts)
    rows = (num_shifts + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case of 1 row

    for i, (y_pred, y_gt) in enumerate(pred_gt_list):
        ax = axes[i]

        y_pred_np = y_pred[:, 0].detach().cpu().numpy()
        y_gt_np = y_gt[:, 0].detach().cpu().numpy()

        min_val = min(y_pred_np.min(), y_gt_np.min())
        max_val = max(y_pred_np.max(), y_gt_np.max())

        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Identity")
        sns.scatterplot(x=y_pred_np, y=y_gt_np, color="blue", alpha=0.5, ax=ax)

        ax.set_xlabel("Estimated Y")
        ax.set_ylabel("Ground-truth Y")
        ax.set_title(f"Shift {i}")
        ax.legend()

    # Remove any unused subplots
    for j in range(len(pred_gt_list), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Log to TensorBoard
    if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
        logger.experiment.add_image(f"{tag}", image_tensor[0], global_step)
        print(f"Logged {tag} with {num_shifts} shifts at step {global_step}")

    plt.close()


def save_causal_effect_plot(y_pred, y_gt, logger, title="Causal Effect Estimation"):
    y_pred, y_gt = y_pred[:, 0].cpu().numpy(), y_gt[:, 0].cpu().numpy()

    min_val = min(y_pred.min(), y_gt.min())
    max_val = max(y_pred.max(), y_gt.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

    sns.scatterplot(x=y_pred, y=y_gt, color="blue", alpha=.5)
    plt.xlabel("Estimated causal effect")
    plt.xlabel("Ground-truth causal effect")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Log to TensorBoard
    if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
        logger.experiment.add_image(f"Causal Effect Estimation", image_tensor[0], 0)
        print(f"Logged Causal Effect Estimation at step 0")
    
    # save locally
    plt.savefig(logger.log_dir + "/causal_effect_estimation.png")
    plt.close()