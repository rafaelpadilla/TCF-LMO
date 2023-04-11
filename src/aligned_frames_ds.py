from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class AlignedFrames(Dataset):
    def __init__(self,
                 ref_frame_paths,
                 tar_frame_paths,
                 transformations = None):

        assert len(ref_frame_paths) != 0, "Empty list ref_frame_paths."
        assert len(tar_frame_paths) != 0, "Empty list tar_frame_paths."
        assert len(ref_frame_paths) == len(tar_frame_paths), f"Lists ref_frame_paths {len(ref_frame_paths)} and tar_frame_paths {len(tar_frame_paths)} must have the same number of frames."

        for frame in ref_frame_paths:
            assert frame.exists(), f"Frame {frame} not found."
        for frame in tar_frame_paths:
            assert frame.exists(), f"Frame {frame} not found."

        self.ref_frame_paths = ref_frame_paths
        self.tar_frame_paths = tar_frame_paths
        self.transformations = transformations

    def __len__(self):
        return len(self.ref_frame_paths) 
    
    def __getitem__(self, idx):
        ref = Image.open(self.ref_frame_paths[idx])
        tar = Image.open(self.tar_frame_paths[idx])

        if self.transformations:
            ref = self.transformations(ref)
            tar = self.transformations(tar)

        return ref, tar, -1, -1

