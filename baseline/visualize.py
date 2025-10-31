import cv2
from matplotlib import pyplot as plt
from pathlib import Path

out_dir = Path("sam_out")

# Pick an example image
mask_img = cv2.imread(str(out_dir / "0_mask.png"), cv2.IMREAD_UNCHANGED)
overlay = cv2.cvtColor(cv2.imread(str(out_dir / "0_overlay.png")), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Label Mask")
plt.imshow(mask_img, cmap="nipy_spectral")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Overlay (Contours)")
plt.imshow(overlay)
plt.axis("off")
plt.show()
