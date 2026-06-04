"""Multimodal processor for LongCat-Next.

Handles image, video, and audio preprocessing for the LongCat-Next model.
Image/video processing uses Qwen2.5-VL's image processor (same visual encoder).
Audio processing uses Whisper-based mel spectrogram extraction.
Video is processed as a sequence of frames through the same visual encoder
with temporal grid_thw = (num_frames, H, W).
"""

import base64
import io
import logging
import tempfile
from typing import List, Union

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


class LongcatNextProcessor(BaseMultimodalProcessor):
    """Processor for LongCat-Next multimodal model."""

    models = [LongcatFlashForCausalLM]

    def __init__(self, hf_config, server_args, processor, transport_mode, **kwargs):
        super().__init__(hf_config, server_args, processor, transport_mode, **kwargs)

        # Get special token IDs from config
        vc = getattr(hf_config, 'visual_config', {})
        ac = getattr(hf_config, 'audio_config', {})
        if isinstance(vc, dict):
            self.image_pad_token_id = vc.get('image_pad_token_id', 131108)
            self.image_start_token_id = vc.get('image_start_token_id', 131106)
            self.image_end_token_id = vc.get('image_end_token_id', 131107)
        else:
            self.image_pad_token_id = getattr(vc, 'image_pad_token_id', 131108)
            self.image_start_token_id = getattr(vc, 'image_start_token_id', 131106)
            self.image_end_token_id = getattr(vc, 'image_end_token_id', 131107)
        if isinstance(ac, dict):
            self.audio_pad_token_id = ac.get('audio_pad_token_id', 131105)
            self.audio_start_token_id = ac.get('audio_start_token_id', 131103)
            self.audio_end_token_id = ac.get('audio_end_token_id', 131104)
        else:
            self.audio_pad_token_id = getattr(ac, 'audio_pad_token_id', 131105)
            self.audio_start_token_id = getattr(ac, 'audio_start_token_id', 131103)
            self.audio_end_token_id = getattr(ac, 'audio_end_token_id', 131104)
        # Use the model's audio_pad_token_id as placeholder
        self._audio_safe_pad = self.audio_pad_token_id  # 131105

        # Set mm_tokens from special tokens and build regex patterns
        self.mm_tokens = self.get_mm_special_tokens()
        self.mm_tokens.build(processor)

        # Initialize audio feature extractor
        self._audio_cfg = ac if isinstance(ac, dict) else {
            k: getattr(ac, k) for k in dir(ac) if not k.startswith('_')
        }

    @staticmethod
    def _clean_input_text(input_text: str) -> str:
        """Clean up input text when chat template stringifies multimodal content.

        When the content is a list like [{'type': 'audio'}, {'type': 'text', 'text': '...'}],
        the chat template's {{ msg.content }} produces a Python repr string.
        Extract just the text portions and reconstruct clean text.
        """
        import ast, re
        # Check if the text contains a stringified content array between chat tokens
        # Pattern: <longcat_user>[{'type': ...}]<longcat_assistant> or similar
        match = re.search(r"(\[{.*?}\])", input_text, re.DOTALL)
        if match:
            try:
                content_list = ast.literal_eval(match.group(1))
                if isinstance(content_list, list):
                    text_parts = []
                    for item in content_list:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                    if text_parts:
                        # Replace the stringified array with just the text
                        clean = input_text[:match.start()] + ' '.join(text_parts) + input_text[match.end():]
                        return clean
            except (ValueError, SyntaxError):
                pass
        return input_text

    @staticmethod
    def get_mm_special_tokens() -> MultimodalSpecialTokens:
        return MultimodalSpecialTokens(
            image_token="<longcat_img_pad>",
            video_token="<longcat_img_pad>",
            audio_token="<longcat_audio_pad>",
        )

    def _process_audio(self, audio_url: str):
        """Process a single audio URL/base64 into mel spectrogram features.

        Returns (mel_features, encoder_length, bridge_length) or None on failure.
        """
        try:
            import numpy as np
            import soundfile as sf

            # Decode base64 or load URL
            if audio_url.startswith('data:'):
                # data:audio/wav;base64,...
                b64_data = audio_url.split(',', 1)[1] if ',' in audio_url else audio_url
                audio_bytes = base64.b64decode(b64_data)
            elif audio_url.startswith('http'):
                import urllib.request
                with urllib.request.urlopen(audio_url) as resp:
                    audio_bytes = resp.read()
            else:
                with open(audio_url, 'rb') as f:
                    audio_bytes = f.read()

            # Load audio with soundfile (present in image; avoids the librosa dep).
            # librosa was only used to decode bytes->waveform; the resample + mel below are unchanged.
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            if data.ndim > 1:  # downmix to mono
                data = data.mean(axis=1)
            waveform = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)  # [1, samples]

            # Resample if needed
            target_sr = self._audio_cfg.get('sampling_rate', 16000)
            if sr != target_sr:
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0),
                    size=int(waveform.shape[1] * target_sr / sr),
                    mode='linear', align_corners=False
                ).squeeze(0)

            # Extract mel spectrogram
            n_fft = self._audio_cfg.get('n_fft', 400)
            hop_length = self._audio_cfg.get('hop_length', 160)
            num_mel_bins = self._audio_cfg.get('num_mel_bins', 128)
            max_audio_seconds = self._audio_cfg.get('max_audio_seconds', 30)

            from transformers.audio_utils import mel_filter_bank
            mel_filters = mel_filter_bank(
                num_frequency_bins=1 + n_fft // 2,
                num_mel_filters=num_mel_bins,
                min_frequency=0.0,
                max_frequency=target_sr / 2.0,
                sampling_rate=target_sr,
                norm="slaney", mel_scale="slaney",
            )

            wave_samples = waveform.shape[1]
            valid_frames = min(max_audio_seconds * target_sr // hop_length,
                            wave_samples // hop_length + 1)

            # Pad to max length
            if wave_samples < max_audio_seconds * target_sr:
                waveform = torch.nn.functional.pad(
                    waveform, (0, max_audio_seconds * target_sr - wave_samples))
            else:
                waveform = waveform[:, :max_audio_seconds * target_sr]

            window = torch.hann_window(n_fft)
            stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2

            mel_spec = torch.from_numpy(mel_filters).float().T @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
            log_spec = (log_spec + 4.0) / 4.0

            mel_features = log_spec[0].numpy()  # [num_mel_bins, time_frames]
            mel_features[:, int(valid_frames):] = 0.0

            # Compute encoder/bridge lengths
            kernel_size = self._audio_cfg.get('kernel_size', 3)
            stride_size = self._audio_cfg.get('stride_size', 2)
            avg_pooler = self._audio_cfg.get('avg_pooler', 4)

            encoder_length = (int(valid_frames) + 2 * (kernel_size // 2) - kernel_size) // 1 + 1
            encoder_length = (encoder_length + 2 * (kernel_size // 2) - kernel_size) // stride_size + 1
            bridge_length = encoder_length // avg_pooler if avg_pooler > 1 else encoder_length

            if bridge_length <= 0:
                return None

            return mel_features, int(encoder_length), int(bridge_length)

        except Exception as e:
            logger.warning(f"Audio processing failed: {e}")
            return None

    def _process_video(self, video_data):
        """Process a video into frames using SGLang's video decoder.

        Returns list of PIL images (frames) or None on failure.
        """
        try:
            from sglang.srt.utils.common import load_video, sample_video_frames

            DESIRED_FPS = 2  # Sample 2 frames per second
            MAX_FRAMES = 64  # Cap at 64 frames

            video = load_video(video_data)
            frame_indices = sample_video_frames(
                video, desired_fps=DESIRED_FPS, max_frames=MAX_FRAMES
            )
            frames_np = video.get_frames_at(frame_indices)  # (N, H, W, C) uint8

            from PIL import Image
            frames = [Image.fromarray(f) for f in frames_np]
            logger.info(f"Video: {len(frames)} frames sampled at {DESIRED_FPS}fps "
                       f"(max {MAX_FRAMES}) from {len(video)} total")
            return frames
        except Exception as e:
            logger.warning(f"Video processing failed: {e}", exc_info=True)
            return None

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        """Process multimodal data for LongCat-Next.

        Handles images, videos, and audio manually without relying on the base
        HF processor, since LongCat's token format differs from Qwen2.5-VL.
        Videos are processed as frame sequences through the same visual encoder.
        """
        # Clean input_text: chat template may stringify content arrays
        input_text = self._clean_input_text(input_text)

        # Tokenize cleaned text directly
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.server_args.model_path)
        input_ids = torch.tensor(tokenizer.encode(input_text), dtype=torch.long)
        mm_items = []

        # Process images: use Qwen2.5-VL image processor for pixel values,
        # then insert framing tokens manually
        has_images = image_data is not None and len(image_data) > 0
        if has_images:
            from transformers import Qwen2VLImageProcessor
            img_proc = Qwen2VLImageProcessor.from_pretrained(self.server_args.model_path)

            # Load images to PIL directly. (SGLang's load_image return type drifted
            # from (PIL, size) to a decoded Tensor across versions, so we don't rely on it.)
            from PIL import Image
            import io, base64, os
            def _to_pil(d):
                if isinstance(d, Image.Image):
                    return d
                if isinstance(d, torch.Tensor):
                    from torchvision.transforms.functional import to_pil_image
                    return to_pil_image(d)
                if isinstance(d, (bytes, bytearray)):
                    return Image.open(io.BytesIO(bytes(d)))
                if isinstance(d, str):
                    if d.startswith('data:'):
                        d = d.split(',', 1)[1]
                        return Image.open(io.BytesIO(base64.b64decode(d)))
                    if d.startswith(('http://', 'https://')):
                        import requests
                        return Image.open(io.BytesIO(requests.get(d, timeout=30).content))
                    if os.path.exists(d):
                        return Image.open(d)
                    return Image.open(io.BytesIO(base64.b64decode(d)))
                raise ValueError(f"Unsupported image_data type: {type(d)}")
            images = []
            for img_data in image_data:
                images.append(_to_pil(img_data).convert('RGB'))

            # Process through Qwen2.5-VL image processor
            processed = img_proc(images=images, return_tensors="pt")
            pixel_values = processed['pixel_values']  # [patches, patch_dim]
            image_grid_thw = processed['image_grid_thw']  # [n_images, 3]

            # Compute number of visual tokens per image
            spatial_merge = getattr(img_proc, 'spatial_merge_size', 2)
            patch_idx = 0
            for i, img in enumerate(images):
                t, h, w = image_grid_thw[i].tolist()
                n_visual_tokens = int(t * (h // spatial_merge) * (w // spatial_merge))

                # Get this image's pixel patches
                n_patches = int(t * h * w)
                img_pixels = pixel_values[patch_idx:patch_idx + n_patches]
                patch_idx += n_patches

                # Create mm_item
                img_item = MultimodalDataItem(modality=Modality.IMAGE)
                img_item.feature = img_pixels
                img_item.image_grid_thw = image_grid_thw[i:i+1]
                img_item.pad_value = self.image_pad_token_id

                # Insert framing tokens: <img_start> <img_pad>×N <img_end>
                img_start_id = getattr(self, 'image_start_token_id', 131106)
                img_end_id = getattr(self, 'image_end_token_id', 131107)
                offset_start = len(input_ids)
                img_frame = torch.tensor(
                    [img_start_id]
                    + [self.image_pad_token_id] * n_visual_tokens
                    + [img_end_id],
                    dtype=input_ids.dtype
                )
                # Insert before <longcat_assistant> token (48)
                assistant_pos = (input_ids == 48).nonzero(as_tuple=True)[0]
                if len(assistant_pos) > 0:
                    pos = assistant_pos[-1].item()
                    input_ids = torch.cat([input_ids[:pos], img_frame, input_ids[pos:]])
                    offset_start = pos
                else:
                    input_ids = torch.cat([input_ids, img_frame])
                    offset_start = len(input_ids) - len(img_frame)

                img_item.offsets = [(offset_start + 1, offset_start + 1 + n_visual_tokens)]
                mm_items.append(img_item)
                logger.info(f"Image {i}: {n_visual_tokens} tokens, grid={image_grid_thw[i].tolist()}, offset={offset_start+1}")

        # Process videos: extract frames, process through same image processor
        video_data = getattr(request_obj, 'video_data', None)
        has_videos = video_data is not None and len(video_data) > 0
        if has_videos:
            from transformers import Qwen2VLImageProcessor
            img_proc = Qwen2VLImageProcessor.from_pretrained(self.server_args.model_path)

            for vid_idx, vid_data in enumerate(video_data):
                frames = self._process_video(vid_data)
                if frames is None or len(frames) == 0:
                    continue

                # Process each frame through Qwen2VLImageProcessor individually,
                # then stack and set temporal grid_thw manually.
                # This is equivalent to video processing but works with all
                # transformers versions that may not support videos= parameter.
                processed = img_proc(images=frames, return_tensors="pt")
                pixel_values = processed['pixel_values']  # all frames' patches
                per_frame_grid = processed['image_grid_thw']  # [n_frames, 3]

                # All frames should have same spatial dims; stack temporally
                # grid_thw for video: (num_frames, H, W) where H,W are per-frame
                n_frames = len(frames)
                _, h_per_frame, w_per_frame = per_frame_grid[0].tolist()
                video_grid_thw = torch.tensor([[n_frames, h_per_frame, w_per_frame]])

                spatial_merge = getattr(img_proc, 'spatial_merge_size', 2)
                n_visual_tokens = int(n_frames * (h_per_frame // spatial_merge) * (w_per_frame // spatial_merge))

                vid_item = MultimodalDataItem(modality=Modality.IMAGE)
                vid_item.feature = pixel_values
                vid_item.image_grid_thw = video_grid_thw
                vid_item.pad_value = self.image_pad_token_id

                # Insert framing tokens
                img_start_id = getattr(self, 'image_start_token_id', 131106)
                img_end_id = getattr(self, 'image_end_token_id', 131107)
                vid_frame = torch.tensor(
                    [img_start_id]
                    + [self.image_pad_token_id] * n_visual_tokens
                    + [img_end_id],
                    dtype=input_ids.dtype
                )
                assistant_pos = (input_ids == 48).nonzero(as_tuple=True)[0]
                if len(assistant_pos) > 0:
                    pos = assistant_pos[-1].item()
                    input_ids = torch.cat([input_ids[:pos], vid_frame, input_ids[pos:]])
                    offset_start = pos
                else:
                    input_ids = torch.cat([input_ids, vid_frame])
                    offset_start = len(input_ids) - len(vid_frame)

                vid_item.offsets = [(offset_start + 1, offset_start + 1 + n_visual_tokens)]
                mm_items.append(vid_item)
                logger.info(f"Video {vid_idx}: {len(frames)} frames, {n_visual_tokens} tokens, "
                           f"grid=({n_frames},{int(h_per_frame)},{int(w_per_frame)})")

        # Process audio: extract mel features, insert pad tokens into text
        if audio_data:
            for audio_url in audio_data:
                result = self._process_audio(audio_url)
                if result is not None:
                    mel_features, encoder_length, bridge_length = result

                    audio_item = MultimodalDataItem(modality=Modality.AUDIO)
                    audio_item.feature = torch.from_numpy(mel_features).float()
                    audio_item.pad_value = self._audio_safe_pad  # safe within vocab range
                    audio_item.model_specific_data = {
                        'encoder_length': encoder_length,
                        'bridge_length': bridge_length,
                    }

                    # Insert framing tokens: <audio_start> <audio_pad>×N <audio_end>
                    offset_start = len(input_ids)
                    audio_frame = torch.tensor(
                        [self.audio_start_token_id]
                        + [self._audio_safe_pad] * bridge_length
                        + [self.audio_end_token_id],
                        dtype=input_ids.dtype
                    )
                    input_ids = torch.cat([input_ids, audio_frame])
                    # Offsets point to the pad region (excluding start/end framing tokens)
                    audio_item.offsets = [(offset_start + 1, offset_start + 1 + bridge_length)]

                    mm_items.append(audio_item)
                    logger.info(f"Audio processed: {bridge_length} tokens, mel={mel_features.shape}, offset={offset_start}")

        # Current SGLang expects a typed MultimodalProcessorOutput, not a dict
        # (the dict return was the old API).
        from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.image_pad_token_id,
            audio_token_id=self._audio_safe_pad,
        )
