import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from decimal import Decimal

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now we can import modules from src
from video import video as VideoModule_Video # Renaming to avoid conflict with local variable 'video'
from chimeric_processor import process as chimeric_process

class TestChimericProcessorInteraction(unittest.TestCase):

    @patch('src.chimeric_processor.video_module.video') # Mock video constructor within chimeric_processor
    @patch('src.chimeric_processor._assemble_chimeric_mkv')
    @patch('src.chimeric_processor._generate_conformed_segments')
    def test_chimeric_files_list_population(self, 
                                             mock_generate_conformed_segments, 
                                             mock_assemble_chimeric_mkv, 
                                             mock_video_constructor_in_processor):
        # 1. Instantiate a REAL `video` object for V_abs_best.
        # We need a real file for the video class constructor to not raise an exception.
        # Let's create a dummy file.
        dummy_vabs_best_path = "dummy_vabs_best.mkv"
        dummy_video_to_conform_path = "dummy_video_to_conform.mkv"
        
        # Create dummy files for VideoModule_Video instantiation
        # Ensure tools.tmpFolder exists or is mocked if VideoModule_Video uses it.
        # For this test, we'll assume VideoModule_Video can handle non-existent files if we mock its methods.
        # However, the constructor itself checks for file existence.
        # So, let's quickly create these dummy files for the test duration.
        
        # To avoid actual file I/O in unit tests as much as possible,
        # we should ideally mock 'tools.file_exists' used by VideoModule_Video constructor.
        
        with patch('src.tools.file_exists', return_value=True):
            V_abs_best_real = VideoModule_Video("dummy_folder", "dummy_vabs_best.mkv")
            # Verify V_abs_best_real.chimeric_files is None initially (as per video.py change)
            self.assertIsNone(V_abs_best_real.chimeric_files, "V_abs_best.chimeric_files should be None after real video init.")
            # The line `if V_abs_best.chimeric_files is None: V_abs_best.chimeric_files = []` in `chimeric_processor.process` handles the initialization to [].

        # Setup Mocks for chimeric_processor.process
        
        # Mock V_abs_best and video_to_conform that are PASSED to process
        # We use the real V_abs_best object created above for the test, as its attribute is being modified.
        # However, for video_to_conform, a mock is fine.
        mock_video_to_conform_instance = MagicMock(spec=VideoModule_Video)
        mock_video_to_conform_instance.filePath = dummy_video_to_conform_path
        mock_video_to_conform_instance.fileBaseName = "dummy_video_to_conform"
        mock_video_to_conform_instance.subtitles = {} # Needed by chimeric_processor
        mock_video_to_conform_instance.audios = {'eng': [{'StreamOrder': '1'}]} # Needed for sync_language

        # This V_abs_best_mock is for the one *passed into* process. We'll use our real one.
        # V_abs_best_real.subtitles and V_abs_best_real.audios need to be set for the function.
        V_abs_best_real.subtitles = {}
        V_abs_best_real.audios = {'eng': [{'StreamOrder': '0'}]}


        current_dict_file_path_obj = {
            V_abs_best_real.filePath: V_abs_best_real, # Use the real V_abs_best
            dummy_video_to_conform_path: mock_video_to_conform_instance
        }
        
        already_compared_dict = {V_abs_best_real.filePath: {dummy_video_to_conform_path: False}} # Indicates V_abs_best is better

        mock_generate_conformed_segments.return_value = (MagicMock(), 'compatible') # Returns (segment_assembly_plan, status)
        
        mock_chimeric_mkv_path = "path/to/chimeric.mkv"
        mock_assemble_chimeric_mkv.return_value = mock_chimeric_mkv_path

        # Mock the video object that is created *inside* chimeric_processor.process
        mock_internal_chimeric_video_obj = MagicMock(spec=VideoModule_Video)
        mock_video_constructor_in_processor.return_value = mock_internal_chimeric_video_obj
        
        # Call the function
        chimeric_process(
            list_not_compatible_video_input_output=[],
            already_compared_dict=already_compared_dict,
            current_dict_file_path_obj=current_dict_file_path_obj,
            common_sync_language='eng',
            length_time_for_initial_offset_calc=10,
            audio_params_for_wav={},
            tools_tmpFolder="mock_tmp",
            audio_params_for_final_encode={}
        )

        # Assertions
        # 1. V_abs_best.chimeric_files is initialized to [] by process
        self.assertIsNotNone(V_abs_best_real.chimeric_files, "V_abs_best.chimeric_files should not be None after process.")
        self.assertIsInstance(V_abs_best_real.chimeric_files, list, "V_abs_best.chimeric_files should be a list.")
        
        # 2. V_abs_best.chimeric_files contains the mocked chimeric_video_obj
        self.assertIn(mock_internal_chimeric_video_obj, V_abs_best_real.chimeric_files, 
                      "V_abs_best.chimeric_files should contain the mocked chimeric_video_obj.")
        
        # Check that the constructor for the internal chimeric video object was called correctly
        mock_video_constructor_in_processor.assert_called_once_with("mock_tmp", os.path.basename(mock_chimeric_mkv_path))
        mock_internal_chimeric_video_obj.get_mediadata.assert_called_once()
        mock_internal_chimeric_video_obj.calculate_md5_streams_split.assert_called_once()


if __name__ == '__main__':
    unittest.main()
