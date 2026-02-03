"""IP-Adapter FaceID for face identity preservation.

Uses ArcFace-based face embeddings (not CLIP) for robust face
identity preservation during audio-reactive transformations.
"""

import torch
import logging
from typing import Optional, List, Tuple
from PIL import Image
from dataclasses import dataclass
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class IPAdapterConfig:
    """Configuration for IP-Adapter FaceID."""

    # IP-Adapter model
    model_id: str = "h94/IP-Adapter-FaceID"
    model_type: str = "ip-adapter-faceid_sd15"  # or ip-adapter-faceid-plus_sd15

    # Face ID settings
    scale: float = 0.6               # Balance identity vs creativity (0.5-0.7 recommended)
    num_tokens: int = 4              # Number of face tokens

    # Face detection
    face_detector: str = "retinaface"  # retinaface, mtcnn, or mediapipe
    face_embedding: str = "arcface"    # ArcFace for robust face ID

    # Audio reactivity
    audio_scale_min: float = 0.3     # Minimum scale during high energy
    audio_scale_max: float = 0.8     # Maximum scale during quiet moments


class FaceDetector:
    """Detects faces and extracts embeddings using ArcFace."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._detector = None
        self._embedding_model = None
        self._initialized = False

    def initialize(self) -> None:
        """Load face detection and embedding models."""
        if self._initialized:
            return

        try:
            # Try insightface for detection + ArcFace embedding
            self._init_insightface()
            self._initialized = True
            logger.info("InsightFace initialized for face detection")

        except ImportError:
            try:
                # Fallback to separate models
                self._init_fallback()
                self._initialized = True
                logger.info("Fallback face detection initialized")
            except Exception as e:
                logger.warning(f"Face detection not available: {e}")

    def _init_insightface(self) -> None:
        """Initialize InsightFace for detection and embedding."""
        from insightface.app import FaceAnalysis

        self._detector = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._detector.prepare(ctx_id=0, det_size=(640, 640))

    def _init_fallback(self) -> None:
        """Initialize fallback face detection."""
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1

            self._detector = MTCNN(
                device=self.device,
                keep_all=True,
                post_process=False,
            )
            self._embedding_model = InceptionResnetV1(
                pretrained='vggface2'
            ).eval().to(self.device)

        except ImportError:
            raise ImportError("No face detection library available")

    def detect_and_embed(
        self,
        image: Image.Image,
    ) -> Tuple[Optional[np.ndarray], Optional[List[dict]]]:
        """Detect faces and extract embeddings.

        Returns:
            Tuple of (embedding array, list of face info dicts)
        """
        if not self._initialized:
            self.initialize()

        if self._detector is None:
            return None, None

        img_array = np.array(image)

        try:
            if hasattr(self._detector, 'get'):
                # InsightFace
                faces = self._detector.get(img_array)
                if not faces:
                    return None, None

                # Get embedding from largest face
                largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                embedding = largest_face.embedding

                face_info = [{
                    'bbox': largest_face.bbox.tolist(),
                    'score': float(largest_face.det_score),
                    'embedding_norm': float(np.linalg.norm(embedding)),
                }]

                return embedding, face_info

            else:
                # MTCNN fallback
                boxes, probs = self._detector.detect(image)
                if boxes is None or len(boxes) == 0:
                    return None, None

                # Get largest face
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                largest_idx = np.argmax(areas)
                box = boxes[largest_idx]

                # Crop and embed
                face = image.crop(box.astype(int))
                face = face.resize((160, 160))
                face_tensor = torch.from_numpy(np.array(face)).float()
                face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
                face_tensor = face_tensor.to(self.device)

                with torch.no_grad():
                    embedding = self._embedding_model(face_tensor).cpu().numpy()[0]

                face_info = [{
                    'bbox': box.tolist(),
                    'score': float(probs[largest_idx]),
                }]

                return embedding, face_info

        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return None, None

    def cleanup(self) -> None:
        """Release resources."""
        self._detector = None
        self._embedding_model = None
        self._initialized = False


class IPAdapterFaceID:
    """IP-Adapter with FaceID for identity preservation."""

    def __init__(
        self,
        config: Optional[IPAdapterConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config or IPAdapterConfig(
            scale=settings.ip_adapter_scale,
        )
        self.device = device
        self.dtype = dtype

        self._face_detector = FaceDetector(device)
        self._ip_adapter = None
        self._image_encoder = None
        self._reference_embedding: Optional[np.ndarray] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize IP-Adapter FaceID."""
        if self._initialized:
            return

        logger.info("Initializing IP-Adapter FaceID...")

        # Initialize face detector
        self._face_detector.initialize()

        # Load IP-Adapter
        self._load_ip_adapter()

        self._initialized = True
        logger.info("IP-Adapter FaceID initialized")

    def _load_ip_adapter(self) -> None:
        """Load IP-Adapter model."""
        try:
            from ip_adapter.ip_adapter_faceid import IPAdapterFaceID as IPAdapterModel

            # Note: Actual loading requires the base pipeline
            # This will be integrated when creating the combined pipeline
            logger.info(f"IP-Adapter model ready: {self.config.model_id}")

        except ImportError as e:
            logger.warning(f"IP-Adapter not available: {e}")

    def set_reference_image(self, image: Image.Image) -> bool:
        """Set reference image for face identity.

        Returns:
            True if face was detected and embedding stored
        """
        if not self._initialized:
            self.initialize()

        embedding, face_info = self._face_detector.detect_and_embed(image)

        if embedding is None:
            logger.warning("No face detected in reference image")
            return False

        self._reference_embedding = embedding
        logger.info(f"Reference face set: {face_info}")
        return True

    def get_embedding(self) -> Optional[np.ndarray]:
        """Get the current reference face embedding."""
        return self._reference_embedding

    def embed_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract face embedding from an image."""
        if not self._initialized:
            self.initialize()

        embedding, _ = self._face_detector.detect_and_embed(image)
        return embedding

    def compute_similarity(
        self,
        image: Image.Image,
    ) -> float:
        """Compute face similarity between image and reference.

        Returns:
            Cosine similarity (0-1, higher = more similar)
        """
        if self._reference_embedding is None:
            return 0.0

        embedding = self.embed_image(image)
        if embedding is None:
            return 0.0

        # Cosine similarity
        ref_norm = self._reference_embedding / np.linalg.norm(self._reference_embedding)
        emb_norm = embedding / np.linalg.norm(embedding)
        similarity = np.dot(ref_norm, emb_norm)

        return max(0.0, float(similarity))

    def get_ip_adapter_kwargs(
        self,
        scale: Optional[float] = None,
    ) -> dict:
        """Get kwargs for IP-Adapter in pipeline.

        Returns:
            Dictionary with ip_adapter_image and scale
        """
        if self._reference_embedding is None:
            return {}

        return {
            'ip_adapter_face_id': self._reference_embedding,
            'ip_adapter_face_id_scale': scale or self.config.scale,
        }

    def apply_to_pipeline(self, pipe):
        """Apply IP-Adapter FaceID to a diffusers pipeline.

        Args:
            pipe: Diffusers pipeline to modify

        Returns:
            Modified pipeline with IP-Adapter
        """
        if not self._initialized:
            self.initialize()

        try:
            from ip_adapter import IPAdapterFaceID as IPAdapterLoader

            # Load IP-Adapter into pipeline
            ip_adapter = IPAdapterLoader(
                pipe,
                self.config.model_id,
                self.device,
            )

            logger.info("IP-Adapter applied to pipeline")
            return ip_adapter

        except Exception as e:
            logger.warning(f"Failed to apply IP-Adapter: {e}")
            return pipe

    def cleanup(self) -> None:
        """Release resources."""
        self._face_detector.cleanup()
        self._ip_adapter = None
        self._image_encoder = None
        self._reference_embedding = None
        self._initialized = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AudioReactiveIPAdapter(IPAdapterFaceID):
    """IP-Adapter with audio-reactive scale modulation.

    Adjusts identity preservation strength based on audio:
    - High energy/beats: Lower scale for more creative freedom
    - Quiet moments: Higher scale for stability
    """

    def __init__(
        self,
        config: Optional[IPAdapterConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(config, device, dtype)
        self._current_scale = self.config.scale

    def update_audio(
        self,
        energy: float,
        is_beat: bool,
        onset_strength: float = 0.0,
    ) -> float:
        """Update scale based on audio state.

        Returns:
            Current effective scale
        """
        base_scale = self.config.scale

        if is_beat:
            # More freedom on beats
            self._current_scale = max(
                self.config.audio_scale_min,
                base_scale * 0.5
            )
        elif onset_strength > 0.5:
            # Moderate freedom on transients
            self._current_scale = max(
                self.config.audio_scale_min,
                base_scale * 0.7
            )
        elif energy > 0.7:
            # Slight reduction for high energy
            self._current_scale = base_scale * 0.8
        elif energy < 0.3:
            # Strengthen during quiet moments
            self._current_scale = min(
                self.config.audio_scale_max,
                base_scale * 1.2
            )
        else:
            # Gradual return to base
            self._current_scale = 0.9 * self._current_scale + 0.1 * base_scale

        return self._current_scale

    def get_ip_adapter_kwargs(
        self,
        scale: Optional[float] = None,
    ) -> dict:
        """Get kwargs with audio-modulated scale."""
        return super().get_ip_adapter_kwargs(
            scale=scale or self._current_scale
        )


# Factory functions
def create_ip_adapter(
    device: str = "cuda",
    audio_reactive: bool = False,
) -> IPAdapterFaceID:
    """Create an IP-Adapter FaceID instance."""
    config = IPAdapterConfig(
        scale=settings.ip_adapter_scale,
    )

    if audio_reactive:
        return AudioReactiveIPAdapter(config, device)
    return IPAdapterFaceID(config, device)
