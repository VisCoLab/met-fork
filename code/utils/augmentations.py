import io

import torch
from PIL import Image, ImageFilter
from torchvision import transforms



def augmentation(key,imsize = 500):
	'''Using ImageNet statistics for normalization.
	'''

	augment_dict = {

		"augment_train":
			transforms.Compose([
				transforms.RandomResizedCrop(imsize, scale=(0.7,1.0),ratio = (0.99,1/0.99)),
				transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
				transforms.RandomGrayscale(p=0.2),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				]),

		"augment_inference":
			transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				]),

		# DINOv3 needs a fixed size divisible by 16; resizing here (square) lets the
		# inference loaders batch (batch>1) instead of the batch_size=1 conv path.
		"augment_inference_resize":
			transforms.Compose([
				transforms.Resize((imsize, imsize)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])

	}

	return augment_dict[key]


# ---------------------------------------------------------------------------
# Phone-photo artifact augmentations (EXP-9).
#
# The Blender renders are "too clean" (EXP-7): they add viewpoint/glass/lighting
# variation but not the degradations that separate a clean render from a real
# gallery phone shot -- JPEG blocking, defocus/shake, ISO grain, resolution loss.
# These transforms inject those artifacts at TRAIN time only, so the recognizer
# learns invariance to them. The inference transform is unchanged (real test
# photos are never augmented).
#
# All custom transforms draw randomness from torch's RNG (not python-random or
# numpy) so they respect torch.manual_seed and PyTorch's per-DataLoader-worker
# seeding -- the existing torchvision transforms (RandomResizedCrop, ...) do the
# same, and this repo sets no worker_init_fn.
# ---------------------------------------------------------------------------

class JPEGCompress:
	"""Re-encode a PIL image as JPEG at a random quality, to mimic phone compression."""

	def __init__(self, qmin=30, qmax=90):
		self.qmin, self.qmax = qmin, qmax

	def __call__(self, img):
		q = int(torch.randint(self.qmin, self.qmax + 1, (1,)).item())
		buf = io.BytesIO()
		img.convert("RGB").save(buf, format="JPEG", quality=q)
		buf.seek(0)
		out = Image.open(buf)
		out.load()                       # decode now, before the BytesIO buffer is freed
		return out.convert("RGB")


class MotionBlur:
	"""Light directional motion blur via a 5x5 normalized line kernel at a random angle."""

	_KERNELS = None

	def __init__(self):
		if MotionBlur._KERNELS is None:
			MotionBlur._KERNELS = self._build()

	@staticmethod
	def _build():
		n, c, kernels = 5, 2, []
		for ang in (0, 45, 90, 135):                       # horizontal / diag / vertical / anti-diag
			k = [[0.0] * n for _ in range(n)]
			for i in range(n):
				if ang == 0:      k[c][i] = 1.0
				elif ang == 90:   k[i][c] = 1.0
				elif ang == 45:   k[n - 1 - i][i] = 1.0
				else:             k[i][i] = 1.0
			flat = [v for row in k for v in row]
			s = sum(flat)
			kernels.append([v / s for v in flat])          # normalize to sum 1 (scale=1.0)
		return kernels

	def __call__(self, img):
		idx = int(torch.randint(0, len(self._KERNELS), (1,)).item())
		return img.convert("RGB").filter(ImageFilter.Kernel((5, 5), self._KERNELS[idx], scale=1.0))


class RandomDownscale:
	"""Downscale by a random factor then upscale back to the original size -> resolution/detail loss."""

	def __init__(self, smin=0.3, smax=0.7):
		self.smin, self.smax = smin, smax

	def __call__(self, img):
		w, h = img.size
		s = self.smin + (self.smax - self.smin) * float(torch.rand(1).item())
		nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
		return img.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)


class AddGaussianNoise:
	"""Additive Gaussian noise on a [0,1] tensor (post-ToTensor, pre-Normalize), clamped to [0,1]."""

	def __init__(self, smin=0.01, smax=0.06):
		self.smin, self.smax = smin, smax

	def __call__(self, t):
		std = self.smin + (self.smax - self.smin) * float(torch.rand(1).item())
		return (t + torch.randn_like(t) * std).clamp_(0.0, 1.0)


# Each phone artifact, wrapped in RandomApply(p=0.5) so the model still sees clean views.
# Built lazily (lambdas) so a fresh instance is created per train transform.
_PHONE = {
	"jpeg":      lambda: transforms.RandomApply([JPEGCompress(30, 90)], p=0.5),                        # PIL
	"blur":      lambda: transforms.RandomApply(
		[transforms.RandomChoice([transforms.GaussianBlur(5, (0.1, 2.0)), MotionBlur()])], p=0.5),     # PIL
	"downscale": lambda: transforms.RandomApply([RandomDownscale(0.3, 0.7)], p=0.5),                   # PIL
	"noise":     lambda: transforms.RandomApply([AddGaussianNoise(0.01, 0.06)], p=0.5),                # tensor
}

# Named training "arms": which phone artifacts to inject. "base" == the original recipe.
ARMS = {
	"base":     [],
	"jpeg":     ["jpeg"],
	"blur":     ["blur"],
	"sensor":   ["downscale", "noise"],                       # ISO noise + resolution loss
	"phoneall": ["jpeg", "blur", "downscale", "noise"],       # all three families stacked
}

_PIL_ORDER = ["downscale", "blur", "jpeg"]                    # physical order: resolution -> optics -> encode
_TENSOR_PHONE = ["noise"]                                    # added after ToTensor, before Normalize


def build_train_transform(imsize=500, arm="base"):
	"""Build the contrastive-training transform for a phone-augmentation `arm`.

	arm="base" returns EXACTLY the original augment_train Compose (no phone artifacts),
	so prior runs reproduce. Other arms insert the artifacts of ARMS[arm] in physical
	order: PIL-stage (downscale -> blur -> jpeg) before ToTensor, tensor-stage (noise)
	after ToTensor and before Normalize.
	"""
	if arm not in ARMS:
		raise ValueError("unknown aug arm {!r}; choices: {}".format(arm, sorted(ARMS)))
	comps = ARMS[arm]
	pil_phone    = [_PHONE[c]() for c in _PIL_ORDER     if c in comps]
	tensor_phone = [_PHONE[c]() for c in _TENSOR_PHONE  if c in comps]
	return transforms.Compose(
		[
			transforms.RandomResizedCrop(imsize, scale=(0.7, 1.0), ratio=(0.99, 1 / 0.99)),
			transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
			transforms.RandomGrayscale(p=0.2),
		]
		+ pil_phone
		+ [transforms.ToTensor()]
		+ tensor_phone
		+ [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
	)
