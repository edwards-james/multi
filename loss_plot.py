import pickle
import matplotlib.pyplot as plt


OUTPUT_DIR = "/home/james/multi"


# load the data
with open(f"{OUTPUT_DIR}/train_stats/train_losses.pkl", "rb") as f:
    train_losses = pickle.load(f)

with open(f"{OUTPUT_DIR}/train_stats/validate_losses.pkl", "rb") as f:
    validate_losses = pickle.load(f)

with open(f"{OUTPUT_DIR}/train_stats/steps.pkl", "rb") as f:
    steps = pickle.load(f)



# make and save a loss plot
fig,ax = plt.subplots(1,1,figsize=(15,5))
ax.plot(steps, train_losses, color="blue", label="train")
ax.plot(steps, validate_losses, color="orange", label="validate")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_ybound(0.9, 1.0)
plt.legend(loc="upper right")
plt.savefig("loss_curve.png", bbox_inches="tight")
print("plot saved to loss_curve.png")