from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from data.loader import DatasetLoader


class DicomLoaderTest(unittest.TestCase):
    def test_loads_a_dicom_series_and_builds_output_affine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            series_uid = generate_uid()
            for index in range(2):
                self._write_slice(root / f"slice-{index}.dcm", series_uid, index)

            dataset = DatasetLoader().load(root)

            self.assertEqual("dicom", dataset.metadata["format"])
            self.assertEqual("ACC001", dataset.studies[0].accession_number)
            series = dataset.studies[0].series[0]
            self.assertEqual((2, 3, 2), series.image.shape)
            self.assertEqual((4, 4), series.affine.shape)
            self.assertTrue(np.array_equal(series.image[:, :, 1], np.full((2, 3), 2)))

    @staticmethod
    def _write_slice(path: Path, series_uid: str, index: int) -> None:
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = MRImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID = series_uid
        dataset.StudyInstanceUID = generate_uid()
        dataset.AccessionNumber = "ACC001"
        dataset.Modality = "MR"
        dataset.SeriesDescription = "T1 enhanced"
        dataset.Rows = 2
        dataset.Columns = 3
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 2.0
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [0, 0, index * 2]
        dataset.InstanceNumber = index + 1
        dataset.PixelData = np.full((2, 3), index + 1, dtype=np.uint16).tobytes()
        dataset.save_as(str(path), enforce_file_format=True)


if __name__ == "__main__":
    unittest.main()
