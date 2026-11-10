from .nodes import PanoramaRandomCrop

NODE_CLASS_MAPPINGS = {
    "PanoramaRandomCropZveroboy": PanoramaRandomCrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaRandomCropZveroboy": "360 Panorama Random Crop (Zveroboy)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']