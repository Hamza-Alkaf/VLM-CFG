"""Qwen3-VL: the Qwen2.5-VL backbone with Qwen3's vision preprocessing."""

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration

from vlmcfg.backbones.qwen2_5_vl import QWEN_MAX_PIXELS, QWEN_MIN_PIXELS, Qwen25VL

MAX_NUM_FRAMES = 32
# The paper caps Qwen3-VL images at Qwen2.5-VL's budget; uncapped, qwen_vl_utils allows
# ~16.8 M pixels, which does not fit a 32 GB GPU on large images.
QWEN3_IMG_MAX_PIXELS = 1605632


class Qwen3VL(Qwen25VL):
    """Qwen3-VL as a guidance backbone.

    mRoPE offsets live on the inner model exactly as in Qwen2.5-VL, so the stream step
    is inherited; only the preprocessing differs: patch 16, `do_resize=False`, and the
    image budget given on each image entry.
    """

    model_cls = Qwen3VLForConditionalGeneration
    base_records_frame = False

    def __init__(
        self,
        pretrained: str,
        img_max_pixels: int = QWEN3_IMG_MAX_PIXELS,
        device_map: str = "auto",
        attn_implementation: str | None = None,
        max_pixels: int = QWEN_MAX_PIXELS,
        min_pixels: int = QWEN_MIN_PIXELS,
        dtype: str = "float16",
    ) -> None:
        self.img_max_pixels = img_max_pixels
        super().__init__(
            pretrained,
            device_map=device_map,
            attn_implementation=attn_implementation,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            dtype=dtype,
        )

    def _image_entry(self, image: Image.Image) -> dict:
        entry = {"type": "image", "image": image}
        if self.img_max_pixels:
            entry["max_pixels"] = self.img_max_pixels
            entry["min_pixels"] = self.min_pixels
        return entry

    def _process(self, batched: list[list[dict]]) -> dict:
        texts = self.processor.apply_chat_template(
            batched, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            batched,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        if videos is not None:
            raise NotImplementedError("video inputs are not supported")
        kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
            "nframes": MAX_NUM_FRAMES,
            **video_kwargs,
        }
        return self.processor(
            text=texts,
            images=images,
            videos=videos,
            **kwargs,
            do_resize=False,
            return_tensors="pt",
        )

    def build_inputs(self, image: Image.Image | None, text: str) -> dict:
        content = [] if image is None else [self._image_entry(image)]
        content.append({"type": "text", "text": text})
        inputs = self._process([[{"role": "user", "content": content}]])
        if image is not None:
            self._record_frame(image.size, inputs)
        return inputs.to(self.input_device)

    def build_chat_inputs(self, messages: list[dict]) -> tuple[list[str], dict]:
        for message in messages:
            for part in message["content"]:
                if part["type"] == "image" and self.img_max_pixels:
                    part.setdefault("max_pixels", self.img_max_pixels)
                    part.setdefault("min_pixels", self.min_pixels)
        batched = [messages]
        texts = self.processor.apply_chat_template(
            batched, tokenize=False, add_generation_prompt=True
        )
        return texts, self._process(batched).to(self.input_device)

    def build_text_only_inputs(self, text: str) -> dict:
        """A text-only prompt through the chat path,
        as the paper's image-free streams used.

        Unlike the conditional inputs, the processor gets no pixel or frame bounds here.
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        texts = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=texts, images=None, videos=None, do_resize=False, return_tensors="pt"
        )
        return inputs.to(self.input_device)


def message_text(messages: list[dict]) -> str | None:
    """The text of the first message (its last text entry),
    as the chat pipeline read it."""
    text = None
    for part in messages[0]["content"]:
        if part["type"] == "text":
            text = part["text"]
    return text
